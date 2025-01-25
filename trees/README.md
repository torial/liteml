## Why

Many implementations (including the base implementation this code was modified from) use numpy.  This version does not use numpy.

For these reasons:
1. To facilitate understanding of the underlying algorithm (ie didactic instead of performant).
2. Minimize external dependencies (the testing portion should be the only portion with external dependencies)
3. Not using numpy also facilitates porting to another language like `V`

## Thanks

The DecisionTree and XGBoost algorithms are modified from the from scratch implementation at https://randomrealizations.com/

