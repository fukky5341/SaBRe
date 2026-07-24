## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 7200 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388)
1: (-26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461)
2: (-24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001)
3: (-28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680)
4: (-31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228)
5: (-28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674)
6: (-55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035)
7: (-32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505)
8: (-37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959)
9: (-30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573)
10: (-49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753)
11: (-48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688)
12: (-59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8373871, 90.8373871)
13: (-51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880)
14: (-79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897)
15: (-38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579)
16: (-48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783)
17: (-79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988)
18: (-48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538)
19: (-38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658)
20: (-34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885)
21: (-46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846)
22: (-49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286)
23: (-37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486)
24: (-45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513)
25: (-39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552)
26: (-56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825)
27: (-46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792)
28: (-37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667)
29: (-51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515)
30: (-46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646)
31: (-49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272)
32: (-55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1568069, 80.1567993)
33: (-73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0983200, 105.0983200)
34: (-63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2917633, 81.2917480)
35: (-60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6587524, 84.6587524)
36: (-60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885986, 86.1885986)
37: (-89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9701691, 107.9701843)
38: (-69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888)
39: (-83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197)
40: (-65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3349915, 87.3349915)
41: (-58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127)
42: (-40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.92 + 160.48 = 163.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -38.2196770, upper bound: 38.2196770

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1335854, upper bound: 38.2164821
time: 79.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2164820, upper bound: 38.1335854
time: 96.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 176.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 176.24
Output dim: 2, lower bound: -38.1335854, upper bound: 38.2164821
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 176.24
Output dim: 2, lower bound: -38.2164820, upper bound: 38.1335854

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8395462, 90.8396606
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1577988, 80.1578522
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0924759, 105.0926819
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2831268, 81.2834625
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6524811, 84.6527252
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885452
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9709778, 107.9710770
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3343887, 87.3344421
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0599758, upper bound: 38.2089764
time: 83.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1262560, upper bound: 38.1434665
time: 76.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8396683, 90.8395538
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1578445, 80.1577988
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0926895, 105.0924683
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2834625, 81.2831268
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6527252, 84.6524887
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885452
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710846, 107.9709702
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344498, 87.3343887
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1434665, upper bound: 38.1262560
time: 90.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2089764, upper bound: 38.0599758
time: 150.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 243.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 243.47
Output dim: 2, lower bound: -38.0599758, upper bound: 38.2089764
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 243.47
Output dim: 2, lower bound: -38.1262560, upper bound: 38.1434665
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 243.47
Output dim: 2, lower bound: -38.1434665, upper bound: 38.1262560
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 243.47
Output dim: 2, lower bound: -38.2089764, upper bound: 38.0599758

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8393936, 90.8396759
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1573257, 80.1574631
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0894394, 105.0900116
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2786713, 81.2795563
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6501923, 84.6507950
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885071, 86.1885300
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9706421, 107.9708939
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3339005, 87.3340454
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9879141, upper bound: 38.2062207
time: 95.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0545660, upper bound: 38.1051677
time: 90.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8395462, 90.8394928
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1577988, 80.1573792
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0924759, 105.0896606
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2831268, 81.2790222
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6524811, 84.6504211
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885223
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9709778, 107.9707336
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3343887, 87.3339539
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0262666, upper bound: 38.1370200
time: 76.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0262666, upper bound: 38.0678835
time: 108.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8394852, 90.8395691
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1573868, 80.1574097
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0896683, 105.0897980
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2790070, 81.2792206
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6504211, 84.6505661
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885300
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9707336, 107.9707947
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3339462, 87.3339920
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0678835, upper bound: 38.1244987
time: 132.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1370200, upper bound: 38.0262666
time: 93.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8396683, 90.8393860
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1578445, 80.1573334
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0926895, 105.0894470
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2834625, 81.2786713
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6527252, 84.6501846
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885376, 86.1885147
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9710846, 107.9706268
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344498, 87.3338928
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1051677, upper bound: 38.0545661
time: 85.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.2062207, upper bound: 37.9879141
time: 91.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 179.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -37.9879141, upper bound: 38.2062207
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.0545660, upper bound: 38.1051677
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.0262666, upper bound: 38.1370200
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.0262666, upper bound: 38.0678835
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.0678835, upper bound: 38.1244987
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.1370200, upper bound: 38.0262666
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.1051677, upper bound: 38.0545661
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 179.87
Output dim: 2, lower bound: -38.2062207, upper bound: 37.9879141

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8452148, 90.8458862
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1602783, 80.1605988
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0726013, 105.0739441
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2601166, 81.2622223
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6361389, 84.6375961
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886368, 86.1886673
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9763947, 107.9770203
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3338394, 87.3342133
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9395699, upper bound: 38.1993650
time: 89.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9762120, upper bound: 38.1411181
time: 145.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8455811, 90.8454895
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1604462, 80.1604004
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0733032, 105.0731659
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2612457, 81.2610016
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6369019, 84.6367569
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886673, 86.1886520
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9767303, 107.9766388
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3340225, 87.3339996
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9395699, upper bound: 38.0984529
time: 192.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9762120, upper bound: 38.0393533
time: 94.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453979, 90.8457108
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1607208, 80.1605072
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0756073, 105.0735931
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2645569, 81.2616882
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6384583, 84.6372223
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886673, 86.1886597
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9767151, 107.9768524
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3343430, 87.3341141
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9780343, upper bound: 38.1301585
time: 90.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0146476, upper bound: 38.0792542
time: 209.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8457336, 90.8453217
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1609039, 80.1603241
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0763245, 105.0728149
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2656860, 81.2604675
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6392212, 84.6363831
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886826, 86.1886444
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9770508, 107.9764786
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3345261, 87.3339005
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9780343, upper bound: 38.0611852
time: 120.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0146476, upper bound: 38.0008974
time: 79.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453064, 90.8457413
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1603241, 80.1605225
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0728149, 105.0736618
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2604675, 81.2617798
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6363831, 84.6372910
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886368, 86.1886673
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9764862, 107.9768829
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3339005, 87.3341293
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0008974, upper bound: 38.1132298
time: 81.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0611852, upper bound: 38.0796600
time: 92.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8457031, 90.8453827
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1605072, 80.1603546
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0735779, 105.0729523
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2616882, 81.2606659
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6372070, 84.6365204
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886673, 86.1886520
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9768524, 107.9765396
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3341141, 87.3339386
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0792542, upper bound: 38.0146476
time: 85.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1301585, upper bound: 37.9780343
time: 95.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8454895, 90.8455658
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1607819, 80.1604462
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0758362, 105.0733109
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2649078, 81.2612457
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6386719, 84.6369171
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886673, 86.1886520
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9768219, 107.9767151
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3344040, 87.3340302
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0393533, upper bound: 38.0430252
time: 108.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0984529, upper bound: 38.0114964
time: 79.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8458862, 90.8452148
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1609650, 80.1602707
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0766144, 105.0726013
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2661285, 81.2601166
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6395264, 84.6361542
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886826, 86.1886368
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9771881, 107.9763794
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3346176, 87.3338470
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0393533, upper bound: 37.9762120
time: 117.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1993650, upper bound: 37.9395699
time: 83.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 203.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9395699, upper bound: 38.1993650
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9762120, upper bound: 38.1411181
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9395699, upper bound: 38.0984529
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9762120, upper bound: 38.0393533
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9780343, upper bound: 38.1301585
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0146476, upper bound: 38.0792542
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -37.9780343, upper bound: 38.0611852
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0146476, upper bound: 38.0008974
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0008974, upper bound: 38.1132298
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0611852, upper bound: 38.0796600
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0792542, upper bound: 38.0146476
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.1301585, upper bound: 37.9780343
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0393533, upper bound: 38.0430252
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0984529, upper bound: 38.0114964
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.0393533, upper bound: 37.9762120
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 203.89
Output dim: 2, lower bound: -38.1993650, upper bound: 37.9395699

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8450012, 90.8460541
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1593170, 80.1598206
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0678406, 105.0699005
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2522888, 81.2555313
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6313782, 84.6336060
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885605, 86.1885986
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9754486, 107.9764481
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3324966, 87.3330460
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9098401, upper bound: 38.1158447
time: 115.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8676510, upper bound: 38.1663128
time: 79.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453674, 90.8456879
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1595001, 80.1596527
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0685577, 105.0691757
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2534180, 81.2543945
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6321716, 84.6328278
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885757, 86.1885834
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9757996, 107.9760971
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3326797, 87.3328476
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9476759, upper bound: 38.0570972
time: 76.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9048529, upper bound: 38.1076112
time: 85.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453674, 90.8456573
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1595001, 80.1596375
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0685425, 105.0691223
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2534027, 81.2543106
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6321564, 84.6327667
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885757, 86.1885834
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9757996, 107.9760742
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3326797, 87.3328323
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9787298, upper bound: 38.0305005
time: 83.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8676510, upper bound: 38.0704257
time: 88.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8457336, 90.8452911
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1596832, 80.1594620
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0692749, 105.0683899
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2545471, 81.2531738
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6329346, 84.6319885
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885910, 86.1885681
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9761505, 107.9757309
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3328629, 87.3326340
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0116512, upper bound: 37.9707201
time: 103.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9644099, upper bound: 38.0106355
time: 85.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8451843, 90.8458710
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1597748, 80.1597443
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0708771, 105.0695496
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2567139, 81.2549820
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6335449, 84.6332321
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885910, 86.1885910
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9757843, 107.9762802
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3329849, 87.3329468
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9500683, upper bound: 38.0469705
time: 89.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0974839
time: 86.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8455505, 90.8455124
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1599579, 80.1595688
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0715942, 105.0688248
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2578583, 81.2538452
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6343079, 84.6324463
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886063, 86.1885757
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9761505, 107.9759293
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3331680, 87.3327560
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9875445, upper bound: 37.9947252
time: 81.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9446083, upper bound: 38.0445802
time: 146.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8455505, 90.8454895
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1599579, 80.1595535
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0715790, 105.0687714
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2578430, 81.2537613
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6343079, 84.6323929
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886063, 86.1885757
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9761505, 107.9759140
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3331680, 87.3327408
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0480846, upper bound: 37.9930901
time: 91.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0337591
time: 80.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8459167, 90.8451233
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1601257, 80.1593781
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0723114, 105.0680466
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2589722, 81.2526245
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6350861, 84.6316147
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886215, 86.1885605
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9764862, 107.9755630
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3333664, 87.3325424
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0825671, upper bound: 37.9316750
time: 87.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0350132, upper bound: 37.9712718
time: 127.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8451233, 90.8459091
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1593781, 80.1597595
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0680542, 105.0696182
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2526245, 81.2550888
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6316223, 84.6333008
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885605, 86.1885910
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9755554, 107.9763107
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3325272, 87.3329697
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9712718, upper bound: 38.0350132
time: 93.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9316749, upper bound: 38.0825671
time: 87.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8454895, 90.8455429
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1595612, 80.1595840
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0687714, 105.0688934
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2537537, 81.2539520
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6324005, 84.6325226
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885757, 86.1885834
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9759064, 107.9759674
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3327408, 87.3327713
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0337591, upper bound: 38.0005356
time: 97.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9930900, upper bound: 38.0480846
time: 89.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8455200, 90.8455505
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1595612, 80.1595917
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0688324, 105.0689087
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2538452, 81.2539673
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6324463, 84.6325378
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885757, 86.1885834
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9759369, 107.9759750
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3327408, 87.3327713
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0445802, upper bound: 37.9446083
time: 85.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9947252, upper bound: 37.9875445
time: 90.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8458862, 90.8451843
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1597443, 80.1594086
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0695496, 105.0681763
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2549744, 81.2528381
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6332397, 84.6317596
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885910, 86.1885681
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9762726, 107.9756241
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3329544, 87.3325729
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0974839, upper bound: 37.9084189
time: 98.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9930900, upper bound: 37.9500683
time: 90.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453064, 90.8457336
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1598358, 80.1596756
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0710907, 105.0692749
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2570496, 81.2545471
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6337585, 84.6329346
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886063, 86.1885834
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9759064, 107.9761429
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3330460, 87.3328705
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0106355, upper bound: 37.9644099
time: 93.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9707201, upper bound: 38.0116512
time: 78.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8456726, 90.8453674
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1600037, 80.1595001
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0718079, 105.0685425
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2581940, 81.2534027
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6345520, 84.6321487
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886063, 86.1885681
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9762421, 107.9757996
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3332291, 87.3326721
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0704257, upper bound: 37.9321588
time: 82.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0305005, upper bound: 37.9787298
time: 90.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8457031, 90.8453751
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1600189, 80.1595001
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0718689, 105.0685577
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2582703, 81.2534256
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6346130, 84.6321640
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886063, 86.1885681
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9762726, 107.9758072
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3332596, 87.3326797
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1076112, upper bound: 37.9048529
time: 101.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.0570972, upper bound: 37.9476759
time: 100.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8460388, 90.8450165
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1602020, 80.1593323
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0725861, 105.0678253
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2594147, 81.2522888
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6353760, 84.6313782
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1886368, 86.1885529
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9766083, 107.9754562
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3334427, 87.3324814
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 664

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1663128, upper bound: 37.8676510
time: 180.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -38.1158447, upper bound: 37.9098401
time: 113.75 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 296.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9098401, upper bound: 38.1158447
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.8676510, upper bound: 38.1663128
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9476759, upper bound: 38.0570972
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9048529, upper bound: 38.1076112
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9787298, upper bound: 38.0305005
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.8676510, upper bound: 38.0704257
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0116512, upper bound: 37.9707201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9644099, upper bound: 38.0106355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9500683, upper bound: 38.0469705
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0974839
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9875445, upper bound: 37.9947252
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9446083, upper bound: 38.0445802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0480846, upper bound: 37.9930901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0337591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0825671, upper bound: 37.9316750
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0350132, upper bound: 37.9712718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9712718, upper bound: 38.0350132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9316749, upper bound: 38.0825671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0337591, upper bound: 38.0005356
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9930900, upper bound: 38.0480846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0445802, upper bound: 37.9446083
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9947252, upper bound: 37.9875445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0974839, upper bound: 37.9084189
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9930900, upper bound: 37.9500683
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0106355, upper bound: 37.9644099
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -37.9707201, upper bound: 38.0116512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0704257, upper bound: 37.9321588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0305005, upper bound: 37.9787298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.1076112, upper bound: 37.9048529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.0570972, upper bound: 37.9476759
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.1663128, upper bound: 37.8676510
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 296.83
Output dim: 2, lower bound: -38.1158447, upper bound: 37.9098401

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8418121, 90.8449860
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1577911, 80.1593018
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0675659, 105.0738754
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2512207, 81.2611313
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6307220, 84.6375198
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884537, 86.1885757
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9730988, 107.9761124
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3321304, 87.3338470
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9009665, upper bound: 38.0570815
time: 105.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8435549, upper bound: 38.1055081
time: 122.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8450012, 90.8428497
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1593170, 80.1582794
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0678406, 105.0696182
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2522888, 81.2544708
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6313782, 84.6329498
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885605, 86.1884995
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9754486, 107.9740753
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3324966, 87.3326874
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8581437, upper bound: 38.1003893
time: 77.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8056374, upper bound: 38.1573764
time: 92.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8421783, 90.8446198
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1579590, 80.1591339
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0682678, 105.0731506
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2523651, 81.2599945
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6315002, 84.6367416
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1884842, 86.1885681
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9734344, 107.9757614
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3323288, 87.3336487
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.9393973, upper bound: 37.9978170
time: 117.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8818080, upper bound: 38.0465844
time: 92.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -57.2282333, 42.9415016, -57.2282333, 42.9415016, -100.1697388, 100.1697388
1: -26.5228977, 35.2737465, -26.5228977, 35.2737465, -61.7966461, 61.7966461
2: -24.3882904, 37.0352097, -24.3882904, 37.0352097, -61.4235001, 61.4235001
3: -28.6646061, 41.4185600, -28.6646061, 41.4185600, -70.0831680, 70.0831680
4: -31.6197968, 41.2804260, -31.6197968, 41.2804260, -72.9002228, 72.9002228
5: -28.0270958, 42.7016716, -28.0270958, 42.7016716, -70.7287674, 70.7287674
6: -55.2653732, 27.2435322, -55.2653732, 27.2435322, -82.5089035, 82.5089035
7: -32.4530487, 40.5343056, -32.4530487, 40.5343056, -72.9873505, 72.9873505
8: -37.1470490, 49.4075470, -37.1470490, 49.4075470, -86.5545959, 86.5545959
9: -30.3097534, 38.3596039, -30.3097534, 38.3596039, -68.6693573, 68.6693573
10: -49.6141319, 48.1668472, -49.6141319, 48.1668472, -97.7809753, 97.7809753
11: -48.5783386, 29.0840302, -48.5783386, 29.0840302, -77.6623688, 77.6623688
12: -59.8568192, 31.3953133, -59.8568192, 31.3953133, -90.8453674, 90.8424835
13: -51.3454437, 47.0207443, -51.3454437, 47.0207443, -98.3661880, 98.3661880
14: -79.6117249, 42.6805649, -79.6117249, 42.6805649, -122.2922897, 122.2922897
15: -38.1548042, 35.1842537, -38.1548042, 35.1842537, -73.3390579, 73.3390579
16: -48.6660004, 37.2414780, -48.6660004, 37.2414780, -85.9074783, 85.9074783
17: -79.5845032, 34.1357956, -79.5845032, 34.1357956, -113.7202988, 113.7202988
18: -48.3018608, 33.4126892, -48.3018608, 33.4126892, -81.7145538, 81.7145538
19: -38.3179092, 19.2692585, -38.3179092, 19.2692585, -57.5871658, 57.5871658
20: -34.7788620, 24.9879265, -34.7788620, 24.9879265, -59.7667885, 59.7667885
21: -46.2488441, 24.9013386, -46.2488441, 24.9013386, -71.1501846, 71.1501846
22: -49.1744423, 25.1914883, -49.1744423, 25.1914883, -74.3659286, 74.3659286
23: -37.8803177, 26.3563347, -37.8803177, 26.3563347, -64.2366486, 64.2366486
24: -45.5024147, 28.8598385, -45.5024147, 28.8598385, -74.3622513, 74.3622513
25: -39.7418480, 29.4632072, -39.7418480, 29.4632072, -69.2050552, 69.2050552
26: -56.0668221, 38.7608604, -56.0668221, 38.7608604, -94.8276825, 94.8276825
27: -46.0897751, 30.1133080, -46.0897751, 30.1133080, -76.2030792, 76.2030792
28: -37.0159988, 29.8611717, -37.0159988, 29.8611717, -66.8771667, 66.8771667
29: -51.2394676, 24.5835857, -51.2394676, 24.5835857, -75.8230515, 75.8230515
30: -46.3585625, 33.4545021, -46.3585625, 33.4545021, -79.8130646, 79.8130646
31: -49.1751671, 27.7795639, -49.1751671, 27.7795639, -76.9547272, 76.9547272
32: -55.6330681, 24.6424675, -55.6330681, 24.6424675, -80.1595001, 80.1581116
33: -73.8043671, 31.8467960, -73.8043671, 31.8467960, -105.0685577, 105.0689011
34: -63.7419434, 17.8972435, -63.7419434, 17.8972435, -81.2534180, 81.2533340
35: -60.8493767, 24.3894367, -60.8493767, 24.3894367, -84.6321716, 84.6321716
36: -60.8835068, 25.3058281, -60.8835068, 25.3058281, -86.1885757, 86.1884842
37: -89.5031738, 18.6231174, -89.5031738, 18.6231174, -107.9757996, 107.9737320
38: -69.7971039, 29.0857887, -69.7971039, 29.0857887, -98.8828888, 98.8828888
39: -83.4528809, 30.7835350, -83.4528809, 30.7835350, -114.2364197, 114.2364197
40: -65.8522110, 21.5000267, -65.8522110, 21.5000267, -87.3326797, 87.3324966
41: -58.7838936, 28.6735172, -58.7838936, 28.6735172, -87.4574127, 87.4574127
42: -40.2477646, 24.7049751, -40.2477646, 24.7049751, -64.9527435, 64.9527435

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8958228, upper bound: 38.0411724
time: 117.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -37.8434457, upper bound: 38.0987127
time: 104.13 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 224.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.9009665, upper bound: 38.0570815
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8435549, upper bound: 38.1055081
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8581437, upper bound: 38.1003893
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8056374, upper bound: 38.1573764
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.9393973, upper bound: 37.9978170
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8818080, upper bound: 38.0465844
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8958228, upper bound: 38.0411724
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 224.15
Output dim: 2, lower bound: -37.8434457, upper bound: 38.0987127
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9787298, upper bound: 38.0305005
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.8676510, upper bound: 38.0704257
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0116512, upper bound: 37.9707201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9644099, upper bound: 38.0106355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9500683, upper bound: 38.0469705
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0974839
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9875445, upper bound: 37.9947252
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9446083, upper bound: 38.0445802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0480846, upper bound: 37.9930901
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9084189, upper bound: 38.0337591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0825671, upper bound: 37.9316750
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0350132, upper bound: 37.9712718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9712718, upper bound: 38.0350132
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9316749, upper bound: 38.0825671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0337591, upper bound: 38.0005356
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9930900, upper bound: 38.0480846
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0445802, upper bound: 37.9446083
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9947252, upper bound: 37.9875445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0974839, upper bound: 37.9084189
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9930900, upper bound: 37.9500683
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0106355, upper bound: 37.9644099
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -37.9707201, upper bound: 38.0116512
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0704257, upper bound: 37.9321588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0305005, upper bound: 37.9787298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.1076112, upper bound: 37.9048529
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.0570972, upper bound: 37.9476759
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.1663128, upper bound: 37.8676510
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 224.15
Output dim: 2, lower bound: -38.1158447, upper bound: 37.9098401

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 163.40 + 7109.99 = 7273.38 seconds
