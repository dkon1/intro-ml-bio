# Deep Learning

## Conceptual

### Question 1

> Consider a neural network with two hidden layers: $p = 4$ input units, 2 units
> in the first hidden layer, 3 units in the second hidden layer, and a single
> output.
>
> a. Draw a picture of the network, similar to Figures 10.1 or 10.4.

<img src="images/nn.png" width="80%" />

> b. Write out an expression for $f(X)$, assuming ReLU activation functions. Be
> as explicit as you can!

The three layers (from our final output layer back to the start of our network)
can be described as:

\begin{align*}
f(X) &= g(w_{0}^{(3)} + \sum^{K_2}_{l=1} w_{l}^{(3)} A_l^{(2)}) \\
A_l^{(2)} &= h_l^{(2)}(X) = g(w_{l0}^{(2)} + \sum_{k=1}^{K_1} w_{lk}^{(2)} A_k^{(1)})\\
A_k^{(1)} &= h_k^{(1)}(X) = g(w_{k0}^{(1)} + \sum_{j=1}^p w_{kj}^{(1)} X_j) \\
\end{align*}

for $l = 1, ..., K_2 = 3$ and $k = 1, ..., K_1 = 2$ and $p = 4$, where,

$$
g(z) = (z)_+ = \begin{cases}
  0, & \text{if } z < 0 \\
  z, & \text{otherwise}
\end{cases}
$$

> c. Now plug in some values for the coefficients and write out the value of
> $f(X)$.

We can perhaps achieve this most easily by fitting a real model. Note,
in the plot shown here, we also include the "bias" or intercept terms.


```r
library(ISLR2)
library(neuralnet)
library(sigmoid)
set.seed(5)
train <- sample(seq_len(nrow(ISLR2::Boston)), nrow(ISLR2::Boston) * 2/3)

net <- neuralnet(crim ~ lstat + medv + ptratio + rm,
    data = ISLR2::Boston[train, ],
    act.fct = relu,
    hidden = c(2, 3)
)
plot(net)
```

We can make a prediction for a given observation using this object.

Firstly, let's find an "ambiguous" test sample


```r
p <- predict(net, ISLR2::Boston[-train, ])
x <- ISLR2::Boston[-train, ][which.min(abs(p - mean(c(max(p), min(p))))), ]
x <- x[, c("lstat", "medv", "ptratio", "rm")]
predict(net, x)
```

```
##         [,1]
## 441 19.14392
```

Or, repeating by "hand":


```r
g <- function(x) ifelse(x > 0, x, 0) # relu activation function
w <- net$weights[[1]] # the estimated weights for each layer
v <- as.numeric(x) # our input predictors

# to calculate our prediction we can take the dot product of our predictors
# (with 1 at the start for the bias term) and our layer weights, lw)
for (lw in w) v <- g(c(1, v) %*% lw)
v
```

```
##          [,1]
## [1,] 19.14392
```

> d. How many parameters are there?


```r
length(unlist(net$weights))
```

```
## [1] 23
```

There are $4*2+2 + 2*3+3 + 3*1+1 = 23$ parameters.

### Question 2

> Consider the _softmax_ function in (10.13) (see also (4.13) on page 141)
> for modeling multinomial probabilities.
>
> a. In (10.13), show that if we add a constant $c$ to each of the $z_l$, then
> the probability is unchanged.

If we add a constant $c$ to each $Z_l$ in equation 10.13 we get:

\begin{align*}
Pr(Y=m|X) 
 &= \frac{e^{Z_m+c}}{\sum_{l=0}^9e^{Z_l+c}} \\
 &= \frac{e^{Z_m}e^c}{\sum_{l=0}^9e^{Z_l}e^c} \\
 &= \frac{e^{Z_m}e^c}{e^c\sum_{l=0}^9e^{Z_l}} \\
 &= \frac{e^{Z_m}}{\sum_{l=0}^9e^{Z_l}} \\
\end{align*}

which is just equation 10.13.

> b. In (4.13), show that if we add constants $c_j$, $j = 0,1,...,p$, to each of
> the corresponding coefficients for each of the classes, then the predictions
> at any new point $x$ are unchanged.

4.13 is 

$$
Pr(Y=k|X=x) = \frac
{e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
{\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}}
$$

adding constants $c_j$ to each class gives:

\begin{align*}
Pr(Y=k|X=x) 
&= \frac
  {e^{\beta_{K0} + \beta_{K1}x_1 + c_1 + ... + \beta_{Kp}x_p + c_p}}
  {\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + c_1 + ... + \beta_{lp}x_p + c_p}} \\
&= \frac
  {e^{c1 + ... + c_p}e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {\sum_{l=1}^K e^{c1 + ... + c_p}e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
&= \frac
  {e^{c1 + ... + c_p}e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {e^{c1 + ... + c_p}\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
&= \frac
  {e^{\beta_{K0} + \beta_{K1}x_1 + ... + \beta_{Kp}x_p}}
  {\sum_{l=1}^K e^{\beta_{l0} + \beta_{l1}x1 + ... + \beta_{lp}x_p}} \\
\end{align*}

which collapses to 4.13 (with the same argument as above).

> This shows that the softmax function is _over-parametrized_. However,
> regularization and SGD typically constrain the solutions so that this is not a
> problem.

### Question 3

> Show that the negative multinomial log-likelihood (10.14) is equivalent to
> the negative log of the likelihood expression (4.5) when there are $M = 2$
> classes.

Equation 10.14 is 

$$
-\sum_{i=1}^n \sum_{m=0}^9 y_{im}\log(f_m(x_i))
$$

Equation 4.5 is:

$$
\ell(\beta_0, \beta_1) = \prod_{i:y_i=1}p(x_i) \prod_{i':y_i'=0}(1-p(x_i'))
$$

So, $\log(\ell)$ is:

\begin{align*}
\log(\ell) 
 &= \log \left( \prod_{i:y_i=1}p(x_i) \prod_{i':y_i'=0}(1-p(x_i')) \right ) \\
 &= \sum_{i:y_1=1}\log(p(x_i)) + \sum_{i':y_i'=0}\log(1-p(x_i')) \\
\end{align*}

If we set $y_i$ to be an indicator variable such that $y_{i1}$ and $y_{i0}$ are
1 and 0 (or 0 and 1) when our $i$th observation is 1 (or 0) respectively, then
we can write:

$$
\log(\ell) = \sum_{i}y_{i1}\log(p(x_i)) + \sum_{i}y_{i0}\log(1-p(x_i'))
$$

If we also let $f_1(x) = p(x)$ and $f_0(x) = 1 - p(x)$ then:

\begin{align*}
\log(\ell) 
 &= \sum_i y_{i1}\log(f_1(x_i)) + \sum_{i}y_{i0}\log(f_0(x_i')) \\
 &= \sum_i \sum_{m=0}^1 y_{im}\log(f_m(x_i)) \\
\end{align*}

When we take the negative of this, it is equivalent to 10.14 for two classes 
($m = 0,1$).

### Question 4

> Consider a CNN that takes in $32 \times 32$ grayscale images and has a single
> convolution layer with three $5 \times 5$ convolution filters (without
> boundary padding).
>
> a. Draw a sketch of the input and first hidden layer similar to Figure 10.8.

<img src="images/nn2.png" width="50%" />

> b. How many parameters are in this model?

There are 5 convolution matrices each with 5x5 weights (plus 5 bias terms) to
estimate, therefore 130 parameters 

> c. Explain how this model can be thought of as an ordinary feed-forward
> neural network with the individual pixels as inputs, and with constraints on
> the weights in the hidden units. What are the constraints?

We can think of a convolution layer as a regularized fully connected layer.
The regularization in this case is due to not all inputs being connected to
all outputs, and weights being shared between connections.

Each output node in the convolved image can be thought of as taking inputs from
a limited number of input pixels (the neighboring pixels), with a set of
weights specified by the convolution layer which are then shared by the
connections to all other output nodes.

> d. If there were no constraints, then how many weights would there be in the
> ordinary feed-forward neural network in (c)?

With no constraints, we would connect each output pixel in our 5x32x32
convolution layer to each node in the 32x32 original image (plus 5 bias terms),
giving a total of 5,242,885 weights to estimate.

### Question 5

> In Table 10.2 on page 433, we see that the ordering of the three methods with
> respect to mean absolute error is different from the ordering with respect to
> test set $R^2$. How can this be?

Mean absolute error considers _absolute_ differences between predictions and 
observed values, whereas $R^2$ considers the (normalized) sum of _squared_
differences, thus larger errors contribute relatively ore to $R^2$ than mean
absolute error.

## Applied

### Question 6

> Consider the simple function $R(\beta) = sin(\beta) + \beta/10$.
>
> a. Draw a graph of this function over the range $\beta \in [−6, 6]$.


```r
r <- function(x) sin(x) + x/10
x <- seq(-6, 6, 0.1)
plot(x, r(x), type = "l")
```

<img src="deep-learning_files/figure-html/unnamed-chunk-7-1.png" width="672" />

> b. What is the derivative of this function?

$$
cos(x) + 1/10
$$

> c. Given $\beta^0 = 2.3$, run gradient descent to find a local minimum of
> $R(\beta)$ using a learning rate of $\rho = 0.1$. Show each of 
> $\beta^0, \beta^1, ...$ in your plot, as well as the final answer.

The derivative of our function, i.e. $cos(x) + 1/10$ gives us the gradient for
a given $x$. For gradient descent, we move $x$ a little in the _opposite_
direction, for some learning rate $\rho = 0.1$:

$$
x^{m+1} = x^m - \rho (cos(x^m) + 1/10)
$$


```r
iter <- function(x, rho) x - rho*(cos(x) + 1/10)
gd <- function(start, rho = 0.1) {
  b <- start
  v <- b
  while(abs(b - iter(b, 0.1)) > 1e-8) {
    b <- iter(b, 0.1)
    v <- c(v, b)
  }
  v
}

res <- gd(2.3)
res[length(res)]
```

```
## [1] 4.612221
```


```r
plot(x, r(x), type = "l")
points(res, r(res), col = "red", pch = 19)
```

<img src="deep-learning_files/figure-html/unnamed-chunk-9-1.png" width="672" />


> d. Repeat with $\beta^0 = 1.4$.


```r
res <- gd(1.4)
res[length(res)]
```

```
## [1] -1.670964
```


```r
plot(x, r(x), type = "l")
points(res, r(res), col = "red", pch = 19)
```

<img src="deep-learning_files/figure-html/unnamed-chunk-11-1.png" width="672" />

### Question 7

> Fit a neural network to the `Default` data. Use a single hidden layer with 10
> units, and dropout regularization. Have a look at Labs 10.9.1–-10.9.2 for
> guidance. Compare the classification performance of your model with that of
> linear logistic regression.





























