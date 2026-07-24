## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 113.951405901
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094)
1: (-49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728)
2: (-48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477)
3: (-52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180)
4: (-65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904)
5: (-55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736)
6: (-86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349)
7: (-67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533)
8: (-78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369)
9: (-61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378)
10: (-91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857)
11: (-83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308)
12: (-60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689)
13: (-67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264)
14: (-116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854)
15: (-66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329)
16: (-97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562)
17: (-109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844)
18: (-91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036)
19: (-67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806)
20: (-65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906)
21: (-84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383)
22: (-74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077)
23: (-66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527)
24: (-85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511)
25: (-63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815)
26: (-90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034)
27: (-99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819)
28: (-66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736)
29: (-78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234)
30: (-83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396)
31: (-89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892)
32: (-76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247)
33: (-113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937)
34: (-90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572)
35: (-88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323)
36: (-87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383)
37: (-136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212)
38: (-109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264)
39: (-121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669)
40: (-112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069)
41: (-86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731)
42: (-58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549)

## BASE Result
execution time: IAR + LP analysis = 2.96 + 155.73 = 158.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -123.2820485, upper bound: 123.2820485


# Binary Search by BASE starts (time budget: 17841.31 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=172.8270263671875
rel_dist={13: [-107.09440620076711, 107.0944062017914]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=172.8270263671875
rel_dist={13: [-109.64944808549234, 109.64944808379073]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=172.8270263671875
rel_dist={13: [-111.95995320337546, 111.95995319383253]}

## Binary Search Result
Binary search time: 508.53 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 17332.78 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1700

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2352134, upper bound: 119.2639978
time: 136.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2639978, upper bound: 119.2352133
time: 109.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 246.22 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 246.22
Output dim: 13, lower bound: -119.2352134, upper bound: 119.2639978
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 246.22
Output dim: 13, lower bound: -119.2639978, upper bound: 119.2352133

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 970

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2229351, upper bound: 119.2619756
time: 163.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2332053, upper bound: 119.2516437
time: 152.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 737

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 838

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2163538, upper bound: 119.1879411
time: 168.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2163244, upper bound: 119.1879704
time: 198.49 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 369.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 369.01
Output dim: 13, lower bound: -119.2229351, upper bound: 119.2619756
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 369.01
Output dim: 13, lower bound: -119.2332053, upper bound: 119.2516437
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 369.01
Output dim: 13, lower bound: -119.2163538, upper bound: 119.1879411
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 369.01
Output dim: 13, lower bound: -119.2163244, upper bound: 119.1879704

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 941

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 531

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2135095, upper bound: 119.2581164
time: 79.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2190651, upper bound: 119.2525724
time: 93.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 827

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2282190, upper bound: 119.2512247
time: 116.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2327820, upper bound: 119.2467115
time: 108.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 562

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1726102, upper bound: 119.1728971
time: 122.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2012648, upper bound: 119.1444385
time: 99.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1831611, upper bound: 119.1686478
time: 183.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1970345, upper bound: 119.1831900
time: 111.53 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 297.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.2135095, upper bound: 119.2581164
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.2190651, upper bound: 119.2525724
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.2282190, upper bound: 119.2512247
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.2327820, upper bound: 119.2467115
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.1726102, upper bound: 119.1728971
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.2012648, upper bound: 119.1444385
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.1831611, upper bound: 119.1686478
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 297.38
Output dim: 13, lower bound: -119.1970345, upper bound: 119.1831900

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 840

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 513

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2130098, upper bound: 119.2523858
time: 434.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2078522, upper bound: 119.2576187
time: 169.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1742

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1727

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1737379, upper bound: 119.2441837
time: 153.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2108737, upper bound: 119.2069469
time: 113.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 709

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1448

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1975915, upper bound: 119.1970862
time: 111.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1742721, upper bound: 119.2198514
time: 201.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1690

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1345003, upper bound: 119.2417598
time: 104.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2277886, upper bound: 119.1481115
time: 109.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1583

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1555095, upper bound: 119.1555441
time: 107.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1555095, upper bound: 119.1555441
time: 148.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1677

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1681

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1817882, upper bound: 119.1430467
time: 106.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.1999547, upper bound: 119.1251690
time: 165.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1370

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2087639, upper bound: 119.1680489
time: 100.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -119.2110509, upper bound: 119.1657436
time: 103.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 206.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2130098, upper bound: 119.2523858
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2078522, upper bound: 119.2576187
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1737379, upper bound: 119.2441837
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2108737, upper bound: 119.2069469
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1975915, upper bound: 119.1970862
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1742721, upper bound: 119.2198514
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1345003, upper bound: 119.2417598
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2277886, upper bound: 119.1481115
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1555095, upper bound: 119.1555441
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1555095, upper bound: 119.1555441
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1817882, upper bound: 119.1430467
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.1999547, upper bound: 119.1251690
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2087639, upper bound: 119.1680489
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 206.19
Output dim: 13, lower bound: -119.2110509, upper bound: 119.1657436
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 206.19
Output dim: 13, lower bound: -119.1970345, upper bound: 119.1831900
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=172.8270263671875
rel_dist={13: [-119.26634416186121, 119.2663441598697]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1724

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1560

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8781789, upper bound: 115.9481937
time: 104.10 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8781789, upper bound: 115.8781789
time: 374.13 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 478.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 478.24
Output dim: 13, lower bound: -115.8781789, upper bound: 115.9481937
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 478.24
Output dim: 13, lower bound: -115.8781789, upper bound: 115.8781789

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 730

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 520

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8772174, upper bound: 115.9338352
time: 108.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8737107, upper bound: 115.9475246
time: 93.45 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 530

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1654

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9444200, upper bound: 115.8715917
time: 107.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9415993, upper bound: 115.8743930
time: 106.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 216.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 216.35
Output dim: 13, lower bound: -115.8772174, upper bound: 115.9338352
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 216.35
Output dim: 13, lower bound: -115.8737107, upper bound: 115.9475246
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 216.35
Output dim: 13, lower bound: -115.9444200, upper bound: 115.8715917
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 216.35
Output dim: 13, lower bound: -115.9415993, upper bound: 115.8743930

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1701

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8499434, upper bound: 115.9293132
time: 212.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8726807, upper bound: 115.9066102
time: 115.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 845

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8146024, upper bound: 115.9448584
time: 132.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8710463, upper bound: 115.8884624
time: 104.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 955

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 748

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8736904, upper bound: 115.8293640
time: 83.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9021935, upper bound: 115.8708833
time: 150.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 608

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1372

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8700469, upper bound: 115.8437098
time: 186.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9109731, upper bound: 115.8728485
time: 99.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 288.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8499434, upper bound: 115.9293132
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8726807, upper bound: 115.9066102
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8146024, upper bound: 115.9448584
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8710463, upper bound: 115.8884624
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8736904, upper bound: 115.8293640
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.9021935, upper bound: 115.8708833
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.8700469, upper bound: 115.8437098
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 288.43
Output dim: 13, lower bound: -115.9109731, upper bound: 115.8728485

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 926

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1684

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8433375, upper bound: 115.9291753
time: 93.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8499434, upper bound: 115.9212510
time: 158.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 737

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8688958, upper bound: 115.9032636
time: 95.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8693268, upper bound: 115.9028297
time: 103.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1748

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7031322, upper bound: 115.9397900
time: 101.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8095571, upper bound: 115.8334012
time: 151.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 521

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8704995, upper bound: 115.8727725
time: 126.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8017090, upper bound: 115.8881858
time: 101.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1633

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9376698, upper bound: 115.8283729
time: 92.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9427212, upper bound: 115.8233287
time: 100.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.7825093, upper bound: 115.8653616
time: 91.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8966617, upper bound: 115.7514230
time: 107.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 724

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9240949, upper bound: 115.8343326
time: 132.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.9306981, upper bound: 115.8277804
time: 176.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1599

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8954507, upper bound: 115.8459057
time: 102.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8841213, upper bound: 115.8573136
time: 112.09 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 216.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8433375, upper bound: 115.9291753
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8499434, upper bound: 115.9212510
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8688958, upper bound: 115.9032636
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8693268, upper bound: 115.9028297
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.7031322, upper bound: 115.9397900
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8095571, upper bound: 115.8334012
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8704995, upper bound: 115.8727725
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8017090, upper bound: 115.8881858
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.9376698, upper bound: 115.8283729
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.9427212, upper bound: 115.8233287
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.7825093, upper bound: 115.8653616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8966617, upper bound: 115.7514230
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.9240949, upper bound: 115.8343326
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.9306981, upper bound: 115.8277804
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8954507, upper bound: 115.8459057
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 216.52
Output dim: 13, lower bound: -115.8841213, upper bound: 115.8573136

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1665

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8422327, upper bound: 115.9268824
time: 98.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -115.8411339, upper bound: 115.9280641
time: 122.08 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 222.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 222.69
Output dim: 13, lower bound: -115.8422327, upper bound: 115.9268824
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 222.69
Output dim: 13, lower bound: -115.8411339, upper bound: 115.9280641
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8499434, upper bound: 115.9212510
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8688958, upper bound: 115.9032636
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8693268, upper bound: 115.9028297
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.7031322, upper bound: 115.9397900
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8095571, upper bound: 115.8334012
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8704995, upper bound: 115.8727725
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8017090, upper bound: 115.8881858
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.9376698, upper bound: 115.8283729
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.9427212, upper bound: 115.8233287
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.7825093, upper bound: 115.8653616
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8966617, upper bound: 115.7514230
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.9240949, upper bound: 115.8343326
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.9306981, upper bound: 115.8277804
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8954507, upper bound: 115.8459057
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.69
Output dim: 13, lower bound: -115.8841213, upper bound: 115.8573136
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=172.8270263671875
rel_dist={13: [-115.95178022674816, 115.95178022664516]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 736

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0453200, upper bound: 114.0533182
time: 127.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0533182, upper bound: 114.0453200
time: 179.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 306.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 306.73
Output dim: 13, lower bound: -114.0453200, upper bound: 114.0533182
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 306.73
Output dim: 13, lower bound: -114.0533182, upper bound: 114.0453200

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1578

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0073744, upper bound: 114.0504296
time: 113.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0073744, upper bound: 114.0153686
time: 166.50 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 526

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0529278, upper bound: 114.0387535
time: 115.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0467433, upper bound: 114.0449287
time: 95.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 212.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 212.95
Output dim: 13, lower bound: -114.0073744, upper bound: 114.0504296
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 212.95
Output dim: 13, lower bound: -114.0073744, upper bound: 114.0153686
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 212.95
Output dim: 13, lower bound: -114.0529278, upper bound: 114.0387535
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 212.95
Output dim: 13, lower bound: -114.0467433, upper bound: 114.0449287

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 771

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0069546, upper bound: 114.0491445
time: 93.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0061035, upper bound: 114.0500052
time: 94.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 1398

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1342

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0424005, upper bound: 114.0046318
time: 109.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9966363, upper bound: 114.0153376
time: 122.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 518

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0527003, upper bound: 114.0337377
time: 141.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0479153, upper bound: 114.0385240
time: 119.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 934

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0440169, upper bound: 114.0429913
time: 149.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0448170, upper bound: 114.0421863
time: 87.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 238.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0069546, upper bound: 114.0491445
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0061035, upper bound: 114.0500052
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0424005, upper bound: 114.0046318
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -113.9966363, upper bound: 114.0153376
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0527003, upper bound: 114.0337377
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0479153, upper bound: 114.0385240
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0440169, upper bound: 114.0429913
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 238.39
Output dim: 13, lower bound: -114.0448170, upper bound: 114.0421863

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 609

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 775

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0065156, upper bound: 114.0377396
time: 94.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9955433, upper bound: 114.0487084
time: 179.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1774

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 823

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0029865, upper bound: 113.9445031
time: 109.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9005748, upper bound: 114.0468945
time: 113.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1617

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0404818, upper bound: 114.0006397
time: 104.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0384741, upper bound: 114.0026960
time: 203.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 657

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -113.9549909, upper bound: 114.0143461
time: 99.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0306749, upper bound: 113.9737398
time: 88.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 528

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0520075, upper bound: 114.0280430
time: 98.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0470331, upper bound: 114.0330464
time: 101.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0460775, upper bound: 114.0105680
time: 107.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0200092, upper bound: 114.0366920
time: 620.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -103.4414444, 67.3344727, -103.4414444, 67.3344727, -170.7759094, 170.7759094
1: -49.9482193, 50.6082573, -49.9482193, 50.6082573, -100.5564651, 100.5564728
2: -48.5611229, 49.9909286, -48.5611229, 49.9909286, -98.5520477, 98.5520477
3: -52.1486244, 63.1560898, -52.1486244, 63.1560898, -115.3047180, 115.3047180
4: -65.5941010, 56.9191933, -65.5941010, 56.9191933, -122.5132904, 122.5132904
5: -55.2361946, 58.8826942, -55.2361946, 58.8826942, -114.1188812, 114.1188736
6: -86.3570557, 50.5043907, -86.3570557, 50.5043907, -136.8614349, 136.8614349
7: -67.3273010, 49.0785522, -67.3273010, 49.0785522, -116.4058533, 116.4058533
8: -78.6864548, 69.6588745, -78.6864548, 69.6588745, -148.3453369, 148.3453369
9: -61.6617050, 67.9820404, -61.6617050, 67.9820404, -129.6437378, 129.6437378
10: -91.6748428, 89.6580505, -91.6748428, 89.6580505, -181.3328705, 181.3328857
11: -83.5685120, 42.3727188, -83.5685120, 42.3727188, -125.9412231, 125.9412308
12: -60.3606529, 76.5750122, -60.3606529, 76.5750122, -136.9356689, 136.9356689
13: -67.9143295, 104.9126892, -67.9143295, 104.9126892, -172.8270264, 172.8270264
14: -116.8467865, 60.6783066, -116.8467865, 60.6783066, -177.5250854, 177.5250854
15: -66.8306656, 63.4330635, -66.8306656, 63.4330635, -130.2637329, 130.2637329
16: -97.4996109, 53.9105530, -97.4996109, 53.9105530, -151.4101562, 151.4101562
17: -109.7044449, 72.3795319, -109.7044449, 72.3795319, -182.0839539, 182.0839844
18: -91.0438004, 46.6898994, -91.0438004, 46.6898994, -137.7337036, 137.7337036
19: -67.0582657, 35.0243301, -67.0582657, 35.0243301, -102.0825882, 102.0825806
20: -65.7333374, 42.6670494, -65.7333374, 42.6670494, -108.4003906, 108.4003906
21: -84.9665527, 42.8776894, -84.9665527, 42.8776894, -127.8442383, 127.8442383
22: -74.9622116, 62.1375923, -74.9622116, 62.1375923, -137.0998077, 137.0998077
23: -66.9117126, 48.2591400, -66.9117126, 48.2591400, -115.1708527, 115.1708527
24: -85.6741333, 55.5482368, -85.6741333, 55.5482368, -141.2223663, 141.2223511
25: -63.0574989, 57.9372826, -63.0574989, 57.9372826, -120.9947815, 120.9947815
26: -90.6047897, 57.4237099, -90.6047897, 57.4237099, -148.0284882, 148.0285034
27: -99.0535889, 46.1071968, -99.0535889, 46.1071968, -145.1607513, 145.1607819
28: -66.5461731, 53.3171005, -66.5461731, 53.3171005, -119.8632736, 119.8632736
29: -78.0615768, 65.4999542, -78.0615768, 65.4999542, -143.5615234, 143.5615234
30: -83.2023392, 55.9533081, -83.2023392, 55.9533081, -139.1556396, 139.1556396
31: -89.7775116, 49.0213776, -89.7775116, 49.0213776, -138.7988892, 138.7988892
32: -76.7950897, 59.4752655, -76.7950897, 59.4752655, -136.2703400, 136.2703247
33: -113.2914581, 70.9033661, -113.2914581, 70.9033661, -184.1947937, 184.1947937
34: -90.1820831, 52.2027702, -90.1820831, 52.2027702, -142.3848572, 142.3848572
35: -88.7005463, 63.8337784, -88.7005463, 63.8337784, -152.5343323, 152.5343323
36: -87.7840805, 60.6282654, -87.7840805, 60.6282654, -148.4123535, 148.4123383
37: -136.1793213, 45.3349075, -136.1793213, 45.3349075, -181.5142212, 181.5142212
38: -109.5075912, 68.6151352, -109.5075912, 68.6151352, -178.1227112, 178.1227264
39: -121.9524078, 68.2549591, -121.9524078, 68.2549591, -190.2073669, 190.2073669
40: -112.7695847, 35.4637299, -112.7695847, 35.4637299, -148.2333069, 148.2333069
41: -86.0540924, 49.4976883, -86.0540924, 49.4976883, -135.5517578, 135.5517731
42: -58.2633934, 46.8866615, -58.2633934, 46.8866615, -105.1500397, 105.1500549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=461, inp2_unstable=461, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=605, inp2_unstable=605, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1342
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 952
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 708
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 903
type: RSZ, layer: 1, pos: 1494
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 968
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1327
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 936
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 951
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 886
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1678

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0392576, upper bound: 114.0412949
time: 138.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -114.0423174, upper bound: 114.0382355
time: 94.67 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 235.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0065156, upper bound: 114.0377396
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -113.9955433, upper bound: 114.0487084
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0029865, upper bound: 113.9445031
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -113.9005748, upper bound: 114.0468945
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0404818, upper bound: 114.0006397
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0384741, upper bound: 114.0026960
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -113.9549909, upper bound: 114.0143461
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0306749, upper bound: 113.9737398
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0520075, upper bound: 114.0280430
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0470331, upper bound: 114.0330464
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0460775, upper bound: 114.0105680
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0200092, upper bound: 114.0366920
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0392576, upper bound: 114.0412949
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 235.25
Output dim: 13, lower bound: -114.0423174, upper bound: 114.0382355
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 235.25
Output dim: 13, lower bound: -114.0448170, upper bound: 114.0421863
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=172.8270263671875
rel_dist={13: [-114.05422291258427, 114.05422290962252]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 12307.98 seconds
