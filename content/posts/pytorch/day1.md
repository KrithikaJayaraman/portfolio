---
author: "Krithika Jayaraman"
linktitle: "Day1"
date: 2020-05-29T16:08:15-07:00
menu:
  main:
    parent: tutorials
next: /tutorials/github-pages-blog
prev: /tutorials/
title: Quick peek about torch.tensor
weight: 10
draft: false
---

## Introduction to Pytorch

Welcome to the first blog in the series Hands-On Pytorch Day 1. Recently I happened to attend a series of web tutorials on "**Pytorch: Zero to GANs**" presented by [Jovian ML](https://www.jovian.ml/). This series is a compilation of the lessons learnt from the course.  

This tutorial will show some of the basic tensor functions in Pytorch. The intent of this blog is to get a feel of how programming in Pytorch looks like. I strongly recommend readers to experiment with the other cool functions available on the official Pytoch page. I assume that you are familiar with basic matrix operations and linear algebra. 

Pytorch is an opensource Machine Learning Library developed by Facebook for Machine Learing, Natural Language Processing and Computer Vision applications.


Let's get hands on some of the basic operations in Pytorch.

Some of the functions that we are experimenting with are

- torch.randn()
- torch.mm()
- torch.clamp()
- torch.clone()
- torch.where()

**Note**: It is assumed that you have Anaconda installed along with all the eseential python libraries such as pandas, numpy, matplotlib, etc.

Before diving into the functions, what is a **tensor**?

A tensor is a data structure which can possibly represent a multi-dimensional object. A 0-D tensor is a scalar, a 1D tensor is a vector, a 2-D tensor is a matrix and so on. At it's core, it's a container for storing an object.

*Source: Deep Learning with Python by Francois Chollet*  
#### **Install torch**

```python
# Import torch and other required modules
import torch
```

Before trying the functions, let's build a couple of tensors.  
```python
x = torch.tensor([1,2,3,4])
print('Number of elements in the tensor: ',x.shape)
print('Dimensionality of the tensor: ',x.ndim)
```
```
Number of elements in the tensor:  torch.Size([4])  
Dimensionality of the tensor:  1
```
In the above example, we have created a one dimensional tensor of size 4. Dimensionality of a tensor here can be thought of the number of axes contained in it.

```python
# tensor from numpy array
import numpy as np
y = torch.tensor(np.array([[[1,1,1,1],
                           [2,2,2,2],
                           [3,3,3,3]],
                          [[1,1,1,1],
                           [2,2,2,2],
                           [3,3,3,3]],
                          [[1,1,1,1],
                           [2,2,2,2],
                           [3,3,3,3]]]))

print('Number of elements in the tensor: ',y.shape)
print('Dimensionality of the tensor: ',y.ndim)
```
```
Number of elements in the tensor:  torch.Size([3, 3, 4])  
Dimensionality of the tensor:  3
```
In this example, we have 3 axes. Assume this as four 3x3 plates stacked behind one another. 

In machine learning and deep learning applications, computing gradients is an essential part of model building and performance tuning process. Pytorch offers a smooth way of setting gradients for specific variables along the model pipeline during the initialization process using the parameter **requires_grad = True**.


```python
#tensor of zeros and ones (typically used in deep learning applications for initializing)
a = torch.ones([3,3], dtype = torch.float32, requires_grad = True)
b = torch.zeros([3,3], dtype = torch.int32)

print(a)
print(b)
```
```
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], requires_grad=True)
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]], dtype=torch.int32)
```



The next example shows how to create a diagonal matrix using pytorch.

```python
# to create a 2D diagonal matrix with 1s on the diagonal
# Function: torch.eye()
c = torch.eye(5)
c
```
```
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]])
```

As part of this exercise, we are also interested in exploring the incorrect usages for better understanding.

What happens if you give requires_grad = True for an integer tensor? Let's check it out.

```python
# What happens if you give requires_grad = True for an integer tensor?

b = torch.ones([2,2], dtype = torch.int32, requires_grad = True)
```
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-5-87f5b3b3c77a> in <module>
      1 # What happens if you give requires_grad = True for an integer tensor?
      2 
----> 3 b = torch.ones([2,2], dtype = torch.int32, requires_grad = True)

RuntimeError: Only Tensors of floating point dtype can require gradients
```
So we have seen what tensors are, how to create them with and without requires_grad = True.

Note that requires_grad is very handy for applications which require the gradient to be calculated. This can be set only for **floating point** tensors.

### Function 1 - torch.randn() and torch.random()  

Creating random variables is a vital part of any simulation or experimental study. In this section, we'll see how to create a random tensor that contains elements from a gaussian distribution with mean 0 and sd 1.

*Function used: torch.randn() torch.randn(size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor*

```python
# Examples

# Create a random 3x3 tensor
w1 = torch.randn([3,3], dtype = torch.float32, requires_grad = True)
print(w1)


# Create 2x2x2 tensor
w2 = torch.randn(2,2,2)
w2
```
```
tensor([[-1.1417,  1.1747,  1.6421],
        [ 2.8420,  0.1834,  0.6659],
        [ 0.8705, -0.3425, -0.1288]], requires_grad=True)
tensor([[[-1.0682,  0.9346],
         [ 0.9322, -0.3019]],

        [[-0.6145, -1.5761],
         [-0.2364, -0.2791]]])
```
When does this code break?

Here since normal distribution is a continuous distribution, the dtype has to be float variants i.e. cannot be integer. However, there's still a workaround. :)

```python
# Create a random 3x3 tensor
w0 = torch.randn([3,3], dtype = torch.int)
print(w0)

```
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-7-ddcc20dd66f7> in <module>
      1 # Create a random 3x3 tensor
----> 2 w0 = torch.randn([3,3], dtype = torch.int)
      3 print(w0)
      4 

RuntimeError: "norma_cpu" not implemented for 'Int'
```
**Workaround**
```python
# Workaround
w0 = torch.randn([3,3]).type(torch.int32)
print(w0)
```
```
tensor([[ 0,  0,  0],
        [ 0,  0,  0],
        [-1,  0,  1]], dtype=torch.int32)
```
Alright! What if we want to define the mean and sd of the normal distribution and then generate random variables?

```python
# use torch.random()
# torch.normal(mean=0.0, std, out=None) → Tensor

w3 = torch.normal(mean = 5, std = 2, size = (2,2))
w3
```
```
tensor([[ 5.0884,  6.4196],
        [11.6668,  2.9319]])
```
One interesting finding here is that for torch.normal, it is possible to specify type and generate integer random variables.  
```python

w4 = torch.normal(mean = 5, std = 2, size = (2,2)).type(torch.int32)
w4
```
```
tensor([[6, 4],
        [5, 2]], dtype=torch.int32)
```
Thus, this is one of the interesting functions that is mostly likely used while modelling. It is also possible to generate random variables from other distributions such as exponential using appropriate functions like torch.exponential() and so on for other distributions.  
### Function 2 - torch.mm()  
torch.mm() is used for matrix multiplication. It's like a sibling to the popular numpy.dot() or numpy.matmul() functions.  
```python# Example 1 
# set the seed to return the same random numbers 
torch.manual_seed(123)

# Two random matrices
x = torch.normal(mean = 5, std = 1, size = (5,3)).type(torch.int32)
w = torch.normal(mean = 2, std = 2,size =(3,2)).type(torch.int32)

print(x,w)
torch.mm(x,w)
```
```
tensor([[4, 5, 4],
        [4, 3, 5],
        [4, 4, 5],
        [4, 5, 4],
        [5, 5, 5]], dtype=torch.int32) tensor([[3, 1],
        [0, 1],
        [1, 1]], dtype=torch.int32)
tensor([[16, 13],
        [17, 12],
        [17, 13],
        [16, 13],
        [20, 15]], dtype=torch.int32)
```
In the above example, x has dimensions 5x3 and w has dimensions 3x5.The torch.mm function performs matrix multiplication of the two tensors.

**What happens if we swap x and w in torch.mm()?**
```python
torch.mm(w,x)
```
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-26-ae41eb236150> in <module>
----> 1 torch.mm(w,x)

RuntimeError: size mismatch, m1: [3 x 2], m2: [5 x 3] at /opt/conda/conda-bld/pytorch_1587428266983/work/aten/src/TH/generic/THTensorMath.cpp:41
```
According to the rules of linear algebra, for matrix multiplication, the number of columns on the first matrix should match the number of rows of the second matrix. Hence caution should be exercised while specifying the dimensions for this operation.

Alright! Let's consider another example as shown below.  
```python
x = torch.randn(5,3)
w = torch.randn(10,3, requires_grad = True)
print(x)
print(w)
```
```
tensor([[-1.9013, -0.9295,  0.5329],
        [-1.1307,  0.1024,  2.5200],
        [-1.2324, -0.8294,  0.4342],
        [-0.7374,  0.1591, -1.3560],
        [ 0.5513,  0.3732,  1.4246]])
tensor([[ 1.2968,  0.6833,  0.2154],
        [ 0.3307, -1.6896, -1.0063],
        [ 0.4166, -1.1895,  0.1669],
        [-1.2620,  0.1699,  1.1698],
        [-0.0621, -1.4270,  1.9431],
        [ 0.2169, -0.1464, -0.4490],
        [-1.8926,  0.6000,  0.3799],
        [-1.8057,  2.9590,  0.8171],
        [ 2.5932, -2.0855, -0.8742],
        [-1.3153, -0.7055, -0.3350]], requires_grad=True)
```
Assuming this is the case in a deep learning application, how do we perform matrix multiplication? Here, we are going to try torch.t() on w matrix.
```python
# Example using torch.t()

torch.mm(x, torch.t(w))
```
```
tensor([[-2.9861,  0.4054,  0.4025,  2.8649,  2.4799, -0.5157,  3.2432,  1.1182,
         -3.4580,  2.9781],
        [-0.8537, -3.0829, -0.1722,  4.3922,  4.8206, -1.3918,  3.1590,  4.4039,
         -5.3487,  0.5708],
        [-2.0714,  0.5568,  0.5456,  1.9222,  2.1037, -0.3409,  1.9998,  0.1260,
         -1.8458,  2.0607],
        [-1.1396,  0.8519, -0.7228, -0.6287, -2.8160,  0.4256,  0.9758,  0.6943,
         -1.0587,  1.3119],
        [ 1.2767, -1.8819,  0.0235,  1.0343,  2.2013, -0.5747, -0.2781,  1.2730,
         -0.5941, -1.4657]], grad_fn=<MmBackward>)
```
The transpose function is nested inside torch.mm() to swap the dimensions of w in order to apply matrix multiplication.

Thus, matrix multiplication is one of the important data manipulations in deep learning. We have also taken a glimpse of the transpose function which goes hand in hand with this operation.
### Function 3 - torch.clamp()  
torch.clamp() is used to bind the random variables within a range. Although I'm not able to think of a real-time use case, I still think it's an interesting function.

*Function: torch.clamp()*  
*torch.clamp(input, min, max, out=None) → Tensor*  

Consider a scenario where we generate random numbers using randperm().

**Usage:** *torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False) → LongTensor*  

```python
# Example using randperm()
#

x = torch.randperm(10)

x
```
```
tensor([0, 9, 1, 4, 3, 8, 5, 6, 7, 2])
```
The above function has generated integer values from 0 to the value 10 (specified as parameter).

Now, assume I changed my mind and I want the lower limit of x to be 4 and upper limit to be 8. We could achieve this using torch.clamp().  
```python
# Example using torch.clamp()

torch.clamp(x, min = 4, max = 8)
```
```
tensor([5, 4, 7, 4, 4, 4, 4, 6, 8, 8])
```
Let's check if we would be able to bind this with floating point values.  
```python
torch.clamp(x, min = 0.2, max = 1.5)
```
```
tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```
Thus, I believe torch.clamp() can be used in statistical analyses to bind variables within a range. For instance, if we want a variable such as 'number of defects' to be positive only and perhaps if we had negative values in the actual data, we could clamp it to have a minimum of 0. It would be a good pre-processing step.

### Function 4 - torch.clone()  
torch.clone() creates a copy of a sensor. The interesting fact about cloning is that the properties of the original tensor doesn't get copied. Only the values get linked. Also, while computing gradient, the copied tensor doesn't get the gradient. Instead, the original tensor takes the gradient even while it is not used in the backward pass.

Let's check out an example.  
```python
x = torch.tensor(np.array([[1,1,1], [2,2,2]]))

y = torch.clone(x)

y
```
```
tensor([[1, 1, 1],
        [2, 2, 2]])
```
Here, y is a clone of the tensor x. Suppose you add a scalar to y. Let's see what happens to x. We will be using torch.add().

*Function: torch.add()*  
*Purpose: to add a scalar to each element in a tensor*  
*torch.add(input, other, out=None)*  
```python
y = torch.add(y,10 )
y
```
```
tensor([[11, 11, 11],
        [12, 12, 12]])
```
```python
# What's in x
x
```
```
tensor([[1, 1, 1],
        [2, 2, 2]])
```
Now let's try simple linear model and apply torch.clone(). Consider a tensor X of size 5x3. Assume it has 5 samples and 3 features x1, x2, x3.

The weight matrix is of dimensions (no. of hidden units, number of input features). Let the number of hidden units be 5.

Let's assume bias to be zero for this example.  
```python
torch.manual_seed(123)
X = torch.randn(5, 3).type(torch.FloatTensor)
y = torch.randn(5, 5).type(torch.float)

# Assume a tensor 'a'
a = torch.randn(5,3, requires_grad = True)


# Clone 'a' to 'w'
w = torch.clone(a)


print('Input X:', X)
print('Output y: ', y)
print('Tensor a:', a)
print('Cloned tensor w: ', w)
```
```
Input X: tensor([[-0.1115,  0.1204, -0.3696],
        [-0.2404, -1.1969,  0.2093],
        [-0.9724, -0.7550,  0.3239],
        [-0.1085,  0.2103, -0.3908],
        [ 0.2350,  0.6653,  0.3528]])
Output y:  tensor([[-1.3250,  0.1784, -2.1338,  1.0524, -0.3885],
        [-0.9343, -0.4991, -1.0867,  0.8805, -2.2685],
        [-0.9133, -0.4204,  1.3111, -0.2199,  0.2190],
        [ 0.2045,  0.5146, -0.2876,  0.8218,  0.1512],
        [ 0.1036, -2.1996, -0.0885, -0.5612,  0.6716]])
Tensor a: tensor([[ 0.9728,  0.0695, -0.4283],
        [ 1.5573,  1.0076,  0.8467],
        [ 1.1068, -0.8800,  0.0642],
        [-0.3424,  0.2524,  0.2091],
        [-1.9297, -0.2152, -0.5500]], requires_grad=True)
Cloned tensor w:  tensor([[ 0.9728,  0.0695, -0.4283],
        [ 1.5573,  1.0076,  0.8467],
        [ 1.1068, -0.8800,  0.0642],
        [-0.3424,  0.2524,  0.2091],
        [-1.9297, -0.2152, -0.5500]], grad_fn=<CloneBackward>)
```
Define the model. Note that tensor 'w' is only used here. Not tensor 'a'.  
```python
learning_rate = 0.001  #Assumed for gradient descent

def model(X, w):
    return torch.matmul(X, torch.t(w))  
```
```python
# Generate predictions
y_pred = model(X, w)

# Compute loss
# Use torch.pow() to compute square
# torch.pow(input, exponent, out=None) → Tensor
loss = (y_pred - y).pow(2).mean()


# Compute loss.backward()
loss.backward()
```
Let's find out the gradient info in w.grad.  
```python
w.grad  
```
```
```
w has no gradient calculated. Alright! Does 'a' have gradient?  
```python
#a.grad.zero_()
a.grad
```
```
tensor([[ -0.0897,  -0.6196,  -0.8324],
        [  5.4647,   8.7851,   2.1213],
        [  1.8918,  -1.7979,  -1.8208],
        [  0.4244,   2.2437,   1.7663],
        [ -5.4859, -11.0232,   0.4795]])
```
The gradient computation has got reflected in 'a'.

Why? 'w' is a copy of 'a' and it doesn't get the 'requires_grad = True' setting while getting cloned. However, it acts as a reference here and the original input gets the gradient through the cloned tensor.  
```python
# Flushing the gradients of a
a.grad.zero_()
```
Thus we have seen an interesting application of torch.clone() to copy tensor elements.  
### Function 5 - torch.where()  
What kind of an experiment could ever skip a filter operation!! Well. That's what torch.where() is for.

*Function: torch.where()*  
*torch.where(condition, x, y) → Tensor*  

The shape of the output tensor is a broadcasted shape of the inputs.

Let's see an example.  
```python
# Example 1 

t1 = torch.normal(mean = -1, std = 4, size = (3,3)).type(torch.int)

t2 = torch.eye(3).type(torch.int)

print(t1)

print(t2)
```
```
tensor([[-2, -3,  0],
        [-1, -1, -2],
        [ 8, -2,  3]], dtype=torch.int32)
tensor([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=torch.int32)
```
```python
torch.where(t1//2 > 0, t1, t2)
```
```
tensor([[1, 0, 0],
        [0, 1, 0],
        [8, 0, 3]], dtype=torch.int32)
```
In the above example, t1 and t2 are two tensors with elements of type 'int'. torch.where() checks if the floor division of t1 by 2 is positive. This happens element-wise. If result is positive, element from t1 is returned in the output, else corresponding element from t2 is returned.  
```python
# Example 2 
t3 = torch.normal(mean = -1, std = 4, size = (3,2)).type(torch.float)

t4 = torch.normal(mean = -1, std = 4, size = ( 3,2)).type(torch.float)

print(t3)

print(t4)
```
```
tensor([[ 4.3571,  4.4551],
        [-1.8060,  9.0302],
        [-4.6036, -3.9113]])
tensor([[-5.4105,  0.8493],
        [-4.1164, -4.1277],
        [-1.6697,  0.4742]])

```
```python 
torch.where((t3+t4) < 0, (t3+2), t4)
```
```
tensor([[ 6.3571,  0.8493],
        [ 0.1940, -4.1277],
        [-2.6036, -1.9113]])
```
Here torch.where() operates upon two integer tensors on the condition (t3+t4) < 0. If this is true, it adds 2 to the element of tensor t3, else gives element from t4.

What happens if the tensors are of different dimensions?  
```python
# Example 3 

t5 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

t6 = torch.normal(mean = -1, std = 4, size = (3,2)).type(torch.int)

print(t5)

print(t6)


torch.where((t5==t6), 1, 0)
```
```
tensor([[[ 1],
         [-1]],

        [[ 4],
         [ 0]],

        [[-4],
         [ 9]]], dtype=torch.int32)
tensor([[ 4, -5],
        [-9, -1],
        [-2, -2]], dtype=torch.int32)
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-82-f239ad21f8a3> in <module>
     10 
     11 
---> 12 torch.where((t5==t6), 1, 0)

/srv/conda/envs/notebook/lib/python3.7/site-packages/torch/tensor.py in wrapped(*args, **kwargs)
     26     def wrapped(*args, **kwargs):
     27         try:
---> 28             return f(*args, **kwargs)
     29         except TypeError:
     30             return NotImplemented

RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```
The above example fails since the shape of the tensors do not match. Let's fix the shape and see.  
```python
t5 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

t6 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

print(t5)

print(t6)


torch.where((t5==t6), t5, t6)
```
```
t5 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

t6 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

print(t5)

print(t6)


torch.where((t5==t6), t5, t6)
```
Now the output has got generated. By the way, what if the datatypes of the tensors differ?  
```python
t7 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.float)

t8 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

print(t7)

print(t8)

torch.where((torch.add(t7,t8)==0), t7, t8)
```
```
tensor([[[-1.0568],
         [-5.9925]],

        [[-0.0270],
         [-2.2952]],

        [[ 6.6277],
         [ 3.5775]]])
tensor([[[-2],
         [ 4]],

        [[-2],
         [-2]],

        [[-1],
         [ 0]]], dtype=torch.int32)
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-90-498b50129220> in <module>
      7 print(t8)
      8 
----> 9 torch.where((torch.add(t7,t8)==0), t7, t8)

RuntimeError: expected scalar type float but found int
```
Evidently, the function expects both the tensors to be of the same datatype. Let's fix and see. 

```python
t7 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

t8 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

print(t7)

print(t8)

torch.where((torch.add(t7,t8)==0), t7, t8)
```
```
tensor([[[ 2],
         [ 0]],

        [[-4],
         [-2]],

        [[-5],
         [-5]]], dtype=torch.int32)
tensor([[[ 1],
         [-6]],

        [[-4],
         [ 2]],

        [[ 1],
         [ 0]]], dtype=torch.int32)
tensor([[[ 1],
         [-6]],

        [[-4],
         [-2]],

        [[ 1],
         [ 0]]], dtype=torch.int32)
```
What if we want a constant as the output?  
```python
t7 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

t8 = torch.normal(mean = -1, std = 4, size = (3,2,1)).type(torch.int)

print(t7)

print(t8)

torch.where((torch.add(t7,t8)==0), 1, t8)
```
```
tensor([[[  4],
         [-10]],

        [[  0],
         [  1]],

        [[ -3],
         [ -5]]], dtype=torch.int32)
tensor([[[ 1],
         [ 1]],

        [[ 2],
         [ 4]],

        [[-3],
         [-3]]], dtype=torch.int32)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-92-d8c185adde4d> in <module>
      7 print(t8)
      8 
----> 9 torch.where((torch.add(t7,t8)==0), 1, t8)

TypeError: where(): argument 'input' (position 2) must be Tensor, not int
```
Unfortunately the function strictly outputs a tensor and doesn't accept constants.

There you go! Key takeways about this torch.where() are

- It requires tensors to be of same datatype and shape

- output needs to be tensor.

### Conclusion
Here we have have quickly seen some of the interesting functions in PyTorch. We'll continue applying these and in fact explore more of these in the next module.

Catch you next week!  

# Reference Links
Official documentation for torch.Tensor: <https://pytorch.org/docs/stable/tensors.html>

