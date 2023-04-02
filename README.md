# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import all the libraries which are needed to the program.
2. get profit prediction graph and computecost value.
3. Get a graph of cost function using gradient descent and also get profit prediction graph.
4. Get the otput of profit for the population of 35,000 and 70,000.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 22005710
RegisterNumber: M.Pranathi
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
    m-len(y)
    h=x.dot(theta)
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err)
    
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
    m=len(y)
    j_history=[]
    for i in range(num_iters):
        predictions=x.dot(theta)
        error=np.dot(x.transpose(),(predictions -y))
        descent=alpha*1/m*error
        theta-=descent
        j_history.append(computeCost(x,y,theta))
    return theta,j_history
    
theta,j_history=gradientDescent(x,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("population of city (10,000s)")
plt.ylabel("profit ($10,000")
plt.title("profit prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population=35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $"+str(round(predict2,0)))

```

## Output:
![linear regression using gradient descent](sam.png)

![profitprediction](https://user-images.githubusercontent.com/118343610/229329646-8977d4ee-b632-4228-b6b6-e96504c5445a.png)


![costcompute](https://user-images.githubusercontent.com/118343610/229329682-b06bbf9a-b82d-4675-80de-59e38d56f04c.png)

![hofx](https://user-images.githubusercontent.com/118343610/229329712-d9a303f0-ef98-4396-8dbb-53f0885aebb2.png)

![cf using gd](https://user-images.githubusercontent.com/118343610/229329727-a976b0c7-7bc9-44f3-8698-e2ca05418f40.png)

![pp](https://user-images.githubusercontent.com/118343610/229329765-bb8e1c08-c7a5-4f2d-b6e1-83ff90782400.png)

![m1](https://user-images.githubusercontent.com/118343610/229329800-d2cfd197-5445-459c-bb60-04d660fe5cdd.png)

![m2](https://user-images.githubusercontent.com/118343610/229329809-e8c68d5f-c7be-43fa-9300-d2d9b77fcc2d.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
