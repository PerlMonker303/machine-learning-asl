<h1>ASL Recognizer</h1>
<h2>Description</h2>
<p>
  ASL stands for "American Sign Language" is a widely used sign language for those requiring hearing and speaking aids. Similar to spoken languages, ASL also uses letters which one may express using one hand and no motion (except for letter Z).
  <br>
  Using my knowledge regarding Deep Learning I trained an Artifical Neural Network to smartly predict ASL signs based on an image input of a hand. The images are of size 28x28 and the data set was taken from <a href="https://www.kaggle.com/datamunge/sign-language-mnist">here</a>.
</p>
<h2>Results</h2>
<p>
  So far I managed to obtain an accuracy of around 67% on the training data and 56% on the development set. Tweaking the 
  regularization parameter, learning rate, number of layers or of hidden units might change these numbers.
</p>
<h2>Concepts practiced</h2>
<ul>
  <li>Multi-Class Classification Problem</li>
  <li>Deep Neural Network</li>
  <li>Multinomial Logistic Regression</li>
  <li>Batch and Mini-Batch Gradient Descent</li>
  <li>Adaptive Moment Estimation (Adam) Optimization</li>
  <li>Feature Scaling with Mean Normalization and Standard Deviation</li>
  <li>Gradient Checking</li>
  <li>L2 Regularization</li>
  <li>He Initialization</li>
  <li>Decaying Learning Rate</li>
  <li>Softmax Layer</li>
</ul>

<h2>Problems I faced</h2>
<ul>
    <li><strong>Vanishing Gradients</strong> - my gradients were going to 0 in the back propagation step although I was using
        the ReLU function for a better performance. I fixed it by scaling the features using the mean normalization and 
        the standard deviation scaling technique
    </li>
    <li><strong>Cost is NaN</strong> - simply decreased the learning rate to fix this as a high learning rate might 
        overshoot and fail to minimize
    </li>
    <li><strong>Gradient Descent not minimizing the cost</strong> - after 8-10 iterations the cost would start increasing.
    To fix this I had to find a smaller learning rate that would fit the optimizer nicely</li>
</ul>

<h2>How to run on your machine and test your own images?</h2>
<ol>
    <li>In the root directory create a directory called "data"</li>
    <li>Inside of it extract the data from the Kaggle link above (in the description)</li>
    <li>Delete everything that is extracted except the two .csv files (sign_mnist_data.csv and sign_mnist_test.csv)</li>
    <li>Create another directory inside "data" called "custom"</li>
    <li>Paste here all the images you want to test yourself - name them in the following manner: "cust_id.jpg" where id 
    will be the index of the image (starting from 1)</li>
    <li>In the "main.py" module change the variable "m_custom" to be equal to the number of custom images you want to 
    run your algorithm on (your own images)</li>
    <li>Run the "main.py" module, wait for the training and at the end you will see in the console your predictions one 
    after the other</li>
</ol>