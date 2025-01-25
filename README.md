# liteml
Starting w/ Karpathy's micrograd implementation, with plans to expand for performance and features.

---
## Commit 4
Added decision tree and simple XGBoost implementations.  Also added initial histogramming support.
[Trees Readme](trees/README.md)

## Commit 3
The commit details below.

## Commit 2
Refactoring of names and organization.

## Commit 1
The following performance optimizations were made based off profiling by `cprofilev`

Original profiled time on my computer: 111.7 seconds.
1. pushed generator expression in loss, cut 6 seconds off run-time (105 seconds, 11.38 for sum)
2. changed Value class to use slots, cut 6 more seconds, (98 seconds)
3. ~~(tried a few things that had no performance value: lamda:None → a static method, localized variables from self)~~
4. removed the set initializer in constructor, cut 10 seconds! (86 seconds)
5. using None (w/ a null check later) instead of Lambda: None, cut 3 seconds (83.5 seconds)
6. ~~using np.multiply added 20 seconds :-O~~
7. ~~converting other objects to slots (no tangible benefit)~~
8. using currying (partial) cut 15 seconds off (68.8 seconds)
9. pushed build_topo to the top, cut 3.3 seconds (65.5 seconds)

----
Initial check-in is based on the video on making [Karpathy's micrograd](https://github.com/karpathy/micrograd), but with performance improvements cutting run-time in half.
