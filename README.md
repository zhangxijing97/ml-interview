# ml-interview

## Supervised Learning

**Definition:** 
A type of machine learning where each training example has both input features **X** and a known output label **y**. The goal is to learn a function \( f: X \rightarrow y \) that can predict labels for new, unseen data.

Common algorithms:
- Linear Regression  
- Logistic Regression  
- Decision Trees / Random Forest  
- Support Vector Machines (SVM)  
- k-Nearest Neighbors (kNN)  
- Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)  
- Neural Networks (for classification/regression)  

## Unsupervised Learning

**Definition:** 
A type of machine learning where the data has only input features **X**, but no labels **y**. The goal is to find hidden patterns, structures, or groupings in the data.

Common algorithms:
- K-Means Clustering  
- Hierarchical Clustering  
- DBSCAN  
- Gaussian Mixture Models (GMM)  
- Principal Component Analysis (PCA)  
- Independent Component Analysis (ICA)  
- Autoencoders  
- t-SNE / UMAP  

## Reinforcement Learning

**Definition:**  
Reinforcement Learning (RL) is a type of machine learning where an **agent** learns by **interacting** with an **environment**.  
At each step:  
1. The agent observes the current **state**.  
2. Chooses an **action** based on its **policy**.  
3. The environment responds with a **reward** and a **next state**.  
4. The agent updates its policy to maximize the total expected future reward.

**Key components:**  
- **Agent** → The learner or decision maker (e.g., a robot, game AI).  
- **Environment** → The world the agent interacts with (e.g., maze, game board).  
- **State** → The current situation of the environment.  
- **Action** → A choice the agent can make.  
- **Reward** → Feedback signal (+/-) guiding the agent toward better behavior.  
- **Policy** → The strategy that maps states to actions.

**Example – Snake Game AI:**  
1. **State:** Snake's position, food position, and current direction.  
2. **Action:** Move up, down, left, or right.  
3. **Reward rules:**  
   - Eat food → +10 points  
   - Hit wall or itself → -100 points  
   - Normal move → -0.1 points (discourages stalling)  
4. **Learning process:**  
   - The agent starts with random moves.  
   - After each move, it gets a reward and sees the new state.  
   - Over many games, it adjusts its po

## What is semi-supervised machine learning?

**Definition:**  
Semi-supervised learning combines supervised and unsupervised learning by training on a small labeled dataset together with a large unlabeled dataset.  

It works under three key assumptions:  
- **Continuity** – Nearby points are likely to share the same label.  
- **Cluster** – Data forms natural clusters, and points in the same cluster share the same label.  
- **Manifold** – Data lies on a lower-dimensional structure, so decision boundaries should follow that shape.  

**Example:**  
Imagine you want to classify cats and dogs. You have 10 labeled photos (5 cats, 5 dogs) and 1,000 unlabeled photos.  
1. Train a model on the 10 labeled images.  
2. Use it to predict labels for the 1,000 unlabeled images.  
3. Keep only high-confidence predictions (e.g., “cat” with 95% probability) and add them to the training set.  
4. Retrain the model, now with much more data, to achieve better accuracy while minimizing labeling cost.  

**Common applications:**  
- Speech recognition  
- Medical image classification  
- Self-driving cars
