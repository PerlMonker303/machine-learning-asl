<h1>ASL Recognizer</h1>
<h2>Description</h2>
<p>
  ASL stands for "American Sign Language" is a widely used sign language for those requiring hearing and speaking aids. Similar to spoken languages, ASL also uses letters which one may express using one hand and no motion (except for letter Z).
  <br>
  Using my knowledge regarding Deep Learning I trained an Artifical Neural Network to smartly predict ASL signs based on an image input of a hand. The images are of size 28x28 and the data set was taken from <a href="https://www.kaggle.com/datamunge/sign-language-mnist">here</a>.
</p>
<h2>Results</h2>
<p>
  TO BE CONCLUDED
</p>
<h2>Concepts practiced</h2>
<ul>
  <li>Multi-Class Classification Problem</li>
  <li>Deep Neural Network</li>
  <li>Batch Gradient Descent</li>
  <li>Feature Scaling with Mean Normalization and Standard Deviation</li>
  <li>Gradient Checking</li>
  <li>L2 Regularization</li>
  <li>He initialization</li>
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
</ul>