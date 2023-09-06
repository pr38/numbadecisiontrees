# numbadecisiontrees
Bellow is a novel 'numba' based recreation of scikit learn decision tree algorithm. It contains the NBDecisionTreeClassifier and NBDecisionTreeRegressor classes, which should function like drop-ins for scikit-learn's DecisionTreeClassifier and DecisionTreeRegressor.


Unlike Sklearn's decision tree, there is no support for multi-column prediction, missing values, sparse arrays, and random splitting. Some criterion and custom criterion are also not supported. But other features are supported; namely class_weights, sample_weights, impurity decrease stoppage, set number of leaf nodes and tree depth, sub-feature selection, minimal split, and minimal leaf sizes.


Unlike Sklearn, numbadecisiontrees's NBDecisionTreeClassifier and NBDecisionTreeRegressor uses numba threads for parallelism for split finding across columns. The number of theads used will be set by ‘numba.set_num_threads’. Based on my rough(and very unscientific)  tests, numbadecisiontrees's single-core split finding performance should match sklearn's. However there is a performance degradation when compared to Sklearn during the whole tree building. I believe that Sklearn is faster when it comes to non-split finding tasks in tree building. One of those tasks may be data-splitting. Sklearn does not create new arrays for each split, but rather rearranges the data, and passes along the two coordinates that node's data is located in between. I do not do this. NBDecisionTreeClassifier and NBDecisionTreeRegressor parallelism should shine when there are a lot of features, when there are a lot of samples(meaning there is more time spent finding split), and there is a limited number of nodes or tree depth. In short, they should be good candidates for boosting.


I should also mention some novel aspects found in the algorithm that Sklearn uses and that I have imitated. When two splits in two different features have the same score approximation, Sklearn will pick one at random. This will often have a ‘butterfly effect’, creating slightly different-looking trees each time. Also, in the case of regression, when examining a split; instead of going back and calculating the true loss, an approximation of sorts is made on sums of each side of the split. 


It should be mentioned that this project was originally an attempt to implement the 'GUIDE tree' algorithm for sklearn/python, then degraded into understanding and trying to match the performance of Sklearn's training time. Hopefully, I now have the foundation to implement novel tree algorithms for sklearn/python. This project was originally architected to have the ability to process multiple nodes in parallel, but when it became clear that the main training loop needed to be numba ‘no-python jited’, due to performance reasons, said ambition  was abandoned. If numba gains support for the threading module, nodes can be processed in parallel.


