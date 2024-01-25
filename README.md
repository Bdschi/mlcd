# mlcd
Machine Learning Algorithm to create Client Dictionary 

## Support Vector Machines and Categorical Values
The input for support vector machines (SVMs) should be numeric values, as SVMs are based on mathematical operations such as dot products and distances.
However, categorical values can be converted to numeric values using various encoding techniques, such as one-hot encoding, ordinal encoding, or hashing encoding [Is it ok to mix categorical and-continuous data for support vector machines?](https://stats.stackexchange.com/questions/50474/is-it-ok-to-mix-categorical-and-continuous-data-for-svm-support-vector-machines), [Support vector machine algorithm](https://www.geeksforgeeks.org/support-vector-machine-algorithm/).
These techniques can transform the categorical values into binary or integer values that can be used by SVMs.
However, some encoding techniques may introduce sparsity, dimensionality, or collinearity issues, which may affect the performance of SVMs. Therefore, it is important to choose the appropriate encoding technique for the data and the problem [How to deal with an svm with categorical attributes?](https://stats.stackexchange.com/questions/52915/how-to-deal-with-an-svm-with-categorical-attributes).
