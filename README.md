<br>
<p align="right">
  <img src="https://user-images.githubusercontent.com/90097422/210276444-d80de34e-69b1-41bb-b33c-d4419f1b6140.png" width="50"><br>
</p>
<br>

# Adapting the Kolmogorov complexity for neural networks

In this repository, you will find the python module <code>neural_network_k_complexity.py</code> that is able to compute an approximation of the Kolmogorov complexity for neural network.

To use it ...

Additionnaly, a micro-study of the possibilities given by this complexity module is also provided

If you want more informations, you can have a look at my article on the subject. Even if some works still remains to be done to better fit an approximation for the K-complexity. the code of this experiment is in the notebook : <code>Adapting the K-complexity for neural network.ipynb</code>
________________________________________________________________________________________________________
## Abstract

*I have come to the conclusion that Kolmogorov complexity is not well adapted to neural networks. You will find in this article an analysis of the different factors that led to this conclusion. Based on this conclusion, it is necessary to find an adaptation of the K-complexity for neural networks. To further our research, we will try to find a computable approximation. In practice, this approximation should be applied to our K-complexity adaptation for neural networks.*  

## Problem  

We will define how the Kolmogorov complexity can be adapted to neural networks. As a reminder, the K-complexity defines the most compact program, in terms of length, that leads to the expected result. It is an evolution of Occam's razor principle. In mathematical terms, it is defined as follows : $C_{U}(s) = min_{p} { |p| : U(p) = s }$, with U a Turing machine, s the expected solution, p a program, and |p| the program length.  
  
We assume that neural networks can be considered as a kind of program. So let us replace p by n, where n is a subset of p, and represent the space of all possible neural networks. We then obtain : $C_{U}(s) = min_{n} { |n| : U(n) = s }$ in which $n \subset p$  

Generally, neural networks are used for classification, so let's replace s by cl, which is a subset of s and represents the classes of a data array. The K-complexity is then written as follows $C_{U}(cl) = min_{n} { |n| : U(n) = cl }$ in which $n \subset p$ and $d \subset s$.  
We now have a less general version of K-complexity, derived directly from the original definition of K-complexity, which applies only to the space of neural networks and the space of neural network solutions.    
However, there are problems with this definition, which can be divided into two parts. First, the neural network provides probabilities instead of definitive answers. Second, neural networks make mistakes and never lead to the exact result expected.  

It is for these reasons that the classical notion of K-complexity \textit{stricto sensus} falters in the case of neural networks. Therefore, we have shown that the K-complexity must be adapted. This will be achieved by redefining the K-complexity to better fit the specificities of neural networks.  

## Method

Let's recap what we have achieved so far. We have found that a version of neural networks of complexity K can be written as follows C_{U}(cl) = min_{n} { |n| : U(n) = cl }$.  

However, this definition has two problems, so let try to adapt it :  
First, the neural network provides probabilities instead of definitive answers, which means that there are multiple answers for a given input. Most of the time, the answer will be the most probable. This means that our equation must be transformed as follows :   $C_{U}(cl) = min_{n} { |n| : max(U(n)) = cl }$ as $U(n)$ becomes a list.  

Second, there is always a percentage of error. This means that novel data networks are not deterministic and sometimes make mistakes. Therefore, they can never achieve exactly the expected result. This leads us to a further modification of the K-complexity :  

$C_{U}(cl) = min_{n} { |n| : max(U(n)) \to cl }$ where the term $max(U(n)) \to s $ represents the extent to which the results of the neural network are close to the solution. The implicit assumption is that this measure can be considered as an approximation of the expected results. This allows us to extend the K-complexity and adapt this fuzzy notion to neural networks.  

We now have a version of the K-complexity that is adapted to neural networks by taking into account the specificities of this type of program. Let's now move on to the computational part. Although it is a great achievement to have a version of K-complexity for neural networks, it is more practical to find a way to approximate it.  

# Experimental protocole  

In essence, the K-complexity captures the efficiency of a program, as it combines the notions of compactness and efficiency. Thus, to get an idea of the efficiency of a neural network, we can compute the ratio between its proximity to the expected results, represented by $max(U(n)) \to cl$ and the size he is taking, which represented by $|n|$. Let's thus approximate the complexity by $C(cl) \approx \frac{max(U(n)) \to cl}{|n|}$. With this formula, we hope that the higher the result, the closer the neural network will be to the ideal K complexity.  

All that remains is to find estimators to compute the two elements of our formulas, the proximity of the expected results and the size of the network.  

The closeness of the expected results can be represented in two ways.  
The most natural estimator would be the number of correct predictions obtained by the network, which corresponds to the accuracy. But it does not take into account all possible error elements.  
Therefore, loss can be a less obvious, but more accurate estimator, as it indicates the difference between the actual and the desired state of the network. However, loss is inversely proportional to accuracy, and if both choices are possible, one of the two must be reversed to preserve the meaning of the results.   

On the other hand, the size of the neural network can be represented in four ways.   
The computational description length of the network can be considered the closest measure of K-complexity. However, it is almost impossible to compute, because much of the code is embedded in built-in functions and is very difficult to access. However, it is almost impossible to compute, because much of the code is nested in built-in functions and is very difficult to access. In addition, we have to break it down to the byte computation phase of the Turing machine, which is also a challenge in itself. For all these reasons, this part will not be implemented in the experiment.  

The number of neurons in the network can give us a quick idea. While this representation can sometimes be useful, it does not take into account how the neurons are connected to each other, and leaves most of the density uncontrolled.   

A more accurate measure would be to examine the number of parameters. This quantity is more useful, as it represents almost all of the information in the network.   

Finally, the size of the memory occupied by the network can also be an interesting measure, as it takes into account some of the more conventional coded components that may be embedded in the network.  

Now that we have many estimators for our quantities, let's create a code that can compute them based on the network information. To get comparable results, some assumptions must be made. As a dataset, we will take the CIFRA-100 small image classification, available on Keras. It has the advantage of being complex enough to allow a wide range of results. After some pre-processing, the data will be fed into the network. The whole network will be composed of a dense layer of 100 neurons with the softmax function as output, which will act as a classifier. Thus, the network composed of this single layer will be taken as our baseline, to evaluate the improvements of the results. All networks will be trained using a batch size of 64, for 20 epochs, with an early stop of one epoch to avoid overfitting.  

The networks used in this experiment will differ in length, ranging from 128 to 4086, in steps of a power of 4, and in width, ranging from 0 to 14, in steps of 2. This evolution does not include the necessary layer, which will be in all networks. These are the dense input layer, size 1024, and the dense classifier, size 100.  

After completing the training and evaluation, the complexity will be calculated using the special module developed to compute the neural network K-complexity approximation. 

## Results

The Figure 1 shows the raw results obtained after the experiment :   

<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/90097422/210274419-b0a9a96d-8eaa-42c6-9520-19eefec6c6c0.png" width="600"><br>
  <b>Figure 1 : Table of row ordered results</b>
</p>
<br>

They were ranked in descending order, because the higher the accuracy or the inverse of the loss, the closer the model is to the expected results. And the higher the complexity, the closer we think we are to the Kolomogorov complexity.  

We observe that all measurements give us the same order for the networks. This is a good indication of the agreement of the chosen estimators, as it shows that they discern the data along similar lines, which is what we wanted. Furthermore, each of our models far outperforms the baseline model. This proves that the baseline is perhaps the most efficient network, using all available metrics, because it has very few connections, very few parameters, tiny memory size, and pretty decent results. Thus, this model won the experiment by far for all complexity metrics.  

In order to further analyze our metrics, a more interpretive version of our results is needed to gain more insight into the relevance of the measures. Therefore, we will standardize our results and plot them in the same graph, as you can see in Figure 2 : 

<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/90097422/210274445-f306f520-f7ad-4b1e-827a-593a096d9693.png" width="600"><br>
  <b>Figure 2 : Graphs of the complexity metrics</b>
</p>
<br>

This graph allows for a better comparison of the metrics used. In this case, the best of all the metrics is the accuracy by neurons, because it highlights the differences by increasing distances better than the other metrics.

## Discussion  

As we have seen in these experiments, there is still a lot of work to be done to adapt K-complexity to neural networks. First, the adapted version we defined at the beginning only holds under certain assumptions. It remains to be seen whether K-complexity can be adapted to a subset of programs. Also, the task performed by the networks must be a classification task.  

For the part where we looked for estimators of our quantities, the most relevant estimator of the program size was discarded. It is really annoying not to use the computational length when fitting the Kolmogorov complexity. Yet, approximations of this quantity have been found. On the other hand, the accuracy seems to be the best available estimator for the proximity of the expected results. This proves that we have achieved some improvement in the estimation of the K-complexity version of the neural network.  

Regarding the results obtained, Figure 3 shows that the model selected by our method has the least accuracy :  

<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/90097422/210274526-4365def6-4bc8-4798-b44d-a8367761534d.png" width="600"><br>
  <b>Figure 3 : Graphs of the accuracy by models</b>
</p>
<br>

This is a big enough problem to invalidate our theory. K-complexity first defines the programs that lead to the expected results, and then takes the shortest one, in that order. The approximation that we have computed instead favors the shortest program, disregarding the accuracy. It now seems clear that our approximation does not work as expected and cannot be considered a K-complexity approximation for neural networks. One possibility to fix this approximation should be by adding coefficents to penalize program length so that the accuracy can be less impactful. But, even if this article does not provide any meaningful estimation, it can be seen as a first draft in the path of getting closer to the desired K-complexity approximation for neural networks.  

Also, significant work remains to be done to analyze the built-in functions and mapping subprocedures to allow the computation of program length in terms of bytes. Overall, although the ideal K-complexity cannot be calculated, this micro-study has shown that approximations can be found, even if the chosen one was not suitable. Ideally, a good approximation should rank programs according to the distance between themselves and the K-complexity. And thus this approximation will have an extremum, certainly infinite, corresponding to the ideal K-complexity, in which it would be possible to get closer.  
