# What is Supervised Machine Learning:
-    Learning a function that maps an input to an ouput (target)
-    Learning is based on example input values with corresponding target values (also called supervisory signals)
    -   E.g. image + object type
-    Tipical usage: predictive modeling
    -   Train model on dataset with input + target values
    -   Use trained model to predict target values for other (new) inputs
-   Classification: target value is class label (discrete attribute, e.g. integer, letter, word)
-   Regression: target value is numerical value (real number)
# Terminology
-   Model : Function, method, with specific parameter values (e.g. a pre-trained neural network)
-   Model class: the class of models (class of methods) in which we search for the model (e.g. NN, SVMs, etc)
-   Parameters: representations of concrete models inside the given model class (e.g. NN)
-   Hyperparameters: parameters controlling model complexity or the training procedure (e.g. network learning rate, the number of hidden layers,etc)
-   Model selection/training: process of finding a model from the model class
# Basic data analysis workflow
-   Question/Task + Data -> Preprocessing -> Choose Features -> Choose Model Class -> Train Model -> Evaluate Model -> Final Model + Answer or Fail
#   Introductory example: Fish Recognition
-   Given: A set of pictures with fish labels
-   Goal: distinguish  between salmons and sea bass

-> Classification task with two labels:
"salmon" (0) vs. "sea bass" (1)
# Preprocessing and Feature Selection
## Feature selection:
-   What data de we have?
-   Removal of redundant features
-   Removal of features the model class cannot utilize
-   (Deep Learning: Feature selection mainly done by neural network)
## Preprocessing:
-   Constrast and brightness correction
-   Segmentation
-   Alignment
-   Normalization
-   ...
# Input representation
-   We can represent each object by a vector of feature values (i.e. feature vectors) of length $d$
$$ x = (x^{(1)},...,x^{(d)})^T$$
Example: each fish is represented as feature vector with two values: $ x^{(1)}$ = length and $x^{(2)}$ = brightness (i.e. $d = 2$)
-   An object described by a feature vector is also referred to as sample
-   Individual $x^{(j)}$ may be:
    -    Group descriptions: categorical variables/ features
    -    Numbers: numerical variables/features (e.g. fish lenght in cm)
-   Assume our dataset consists of $l$ objects with feature vectors $x_1,..., x_l$
-   Each feature vector is of length $d$
-   Then we can write the feature vectors of all objects in a matrix of feature vectors $X$:
$$ X = (x_1,..,x_l) = \begin{bmatrix} x_1^{(1)} & ... & x_l^{(1)}\\ \vdots & \vdots & \vdots \\ x_1^{(d)} & ... & x_l^{(d)} \end{bmatrix}$$
-   $x_i^{(j)}$ is the $j$-th feature value of fish $i$
# Input and output representation
-   Assume we are given a target value $y_i$ for each sample $x_i$
-   Then all target values constitute the target/label vector:
$$ y = (y_i,...,y_l)^T$$
- Often we write our dataset, including input features and targets, as data matrix $Z$:
$$Z = \begin{bmatrix} X \\ y^T \end{bmatrix} = \begin{bmatrix} x_1^{(1)} & ... & x_l^{(1)}\\ \vdots & \vdots & \vdots \\ x_1^{(d)} & ... & x_l^{(d)} \\ y_1 & ... & y_l \end{bmatrix}$$
-   Note that if a vector of target values is given for each sample, then we get a target value matrix $Y$

#   How do we get the "best" model?
-   How does our model perform on our data? - Loss function
-   How will it perform on (unseen) future data? I.e. how will it generalize?
#   Scoring our models: Loss function
-   Assume we have a model $g$, parameterized by $w$
-   $g(x;w)$  maps an input vector x to an output value $\hat{y}$
-   We want (prediction) $\hat{y}$ to be as close as possible to the true target value $y$
-   We can use a loss function
$$ L(y,g(x;w))$$
 to measure how close our prediction is to the true target for a given sample with $z = (X^T,y)^T$
 -  The smaller the lost (cost), the better our prediction
 #  Examples of loss functions
 -  Zero-one loss:

 $L_{zo}(y,g(x;w)) = $
 -  0 if $y = g(x;w)$
 -  1 if $y \ne g(x;w)$

 -  Quadratic loss:

 $L_{q}(y,g(x;w)) = (y -g(x;w))^2 $

 -  Many other loss functions available with different justifications
 -  Not every loss function is suitable for every task
 -  Choice of loss function depends on data, task, and model class
 # Generalization error/risk
 How will it perform on (unseen) future data? - Generalization error (risk)
 -  The generalization error or risk is the expected loss on future data for a given model $g(.;w)$:

$$R(g(.;w)) = E_Z[L(y,g(x;w))] = \int_X\int_R L(y,g(x;w))p(x,y)dydx$$
-   $R(g(x;w)) = E_{y|x}[L(y,g(x;w))]$ denotes the expected loss for input x (integration only over y)
-   Inpractice, we hardly have any knowledge about $p(x,y)$
-   -> We have to estimate the generalization error
# Empirical Risk Minimization (ERM)
-   We do not know the true $p(x,y)$ but we have access to a subset of l data samples (i.e. our dataset)
-   We estimate the (true) risk by the empirical risk $R_{emp}$ on our dataset:
$$$$
 

