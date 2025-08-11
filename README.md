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

## How to Choose Which Algorithm to Use for a Dataset

Choosing an algorithm depends on both the **dataset** and the **business/application requirements**. You can apply supervised and unsupervised learning to the same data.

**General guidelines:**
- **Supervised learning** → Requires **labeled data**.  
  - **Regression** → Target is continuous numerical values (e.g., price, temperature).  
  - **Classification** → Target is categorical (e.g., spam/ham, cat/dog).  
- **Unsupervised learning** → Requires **unlabeled data** (e.g., clustering, dimensionality reduction).  
- **Semi-supervised learning** → Requires a combination of **labeled and unlabeled data**.  
- **Reinforcement learning** → Requires an **environment**, **agent**, **state**, and **reward** for learning through interaction.

## K Nearest Neighbor (KNN)

**Definition:**  
KNN is a **supervised**, **non-parametric** algorithm used for **classification** and **regression**.  
It predicts based on the labels of the **K closest data points** in the training set, without assuming any data distribution.

**How it works (classification):**  
1. Choose **K** (e.g., K=5).  
2. Calculate the **distance** (commonly Euclidean) from the new point to all labeled points.  
3. Select the **K nearest neighbors**.  
4. Assign the label by **majority vote** among the K neighbors.  

**Example:**  
If K=5 and the 5 nearest neighbors include **3 red** and **2 green**, the new point is labeled **red**.

## Feature Importance in Machine Learning

**Definition:**  
Feature importance measures how much each input feature contributes to predicting the target variable.  
It helps us:  
- Identify which features are most useful  
- Understand data structure and model behavior  
- Simplify the model and improve interpretability  

## Feature Importance in Machine Learning

**Definition:**  
Feature importance measures how much each input feature contributes to predicting the target variable.  
It helps us:  
- Identify which features are most useful  
- Understand data structure and model behavior  
- Simplify the model and improve interpretability  

**Common methods to determine feature importance:**

1. **Model-based Importance**  
   - Some models (e.g., Decision Tree, Random Forest) have built-in importance scores.  
   - Example: Random Forest calculates the **decrease in node impurity** (Gini/entropy) weighted by the probability of reaching that node, averaged over all trees.  

2. **Permutation Importance**  
   - Shuffle a feature’s values in the validation set and measure the drop in model performance.  
   - A larger drop means the feature is more important.  

3. **SHAP (SHapley Additive Explanations)**  
   - Based on game theory, measures each feature’s contribution to a single prediction.  
   - Works well for complex models like gradient boosting or neural networks.  

4. **Correlation Coefficients**  
   - Compute correlation (e.g., Pearson, Spearman) between a feature and the target variable.  
   - Useful for quick checks, but only captures linear or monotonic relationships.  

**Uses of feature importance:**
- **Model optimization:** Remove low-importance features to reduce overfitting.  
- **Interpretability:** Explain model decisions to stakeholders.  
- **Data analysis:** Identify factors most related to the target.

## Overfitting in Machine Learning

**Definition:**  
Overfitting occurs when a model performs well on training data but fails to generalize to unseen data because it has **memorized the training data** instead of learning the underlying patterns.

**How to avoid overfitting:**
1. **Cross-validation** – Evaluate the model on multiple train/validation splits to ensure generalization.  
2. **Regularization (L1, L2)** – Add penalty terms to the loss function to prevent overfitting model parameters.  
3. **Reduce model complexity** – Use simpler models or limit parameters (e.g., prune decision trees, reduce neural network layers).  
4. **Collect more data / Data augmentation** – Increase training data volume or synthetically generate variations to improve robustness.

## Overfitting in Machine Learning

**Definition:**  
Overfitting occurs when a model performs well on training data but fails to generalize to unseen data because it has **memorized the training data** instead of learning the underlying patterns.

**How to avoid overfitting:**
1. **Cross-validation** – Evaluate the model on multiple train/validation splits to ensure generalization.  
2. **Regularization (L1, L2)** – Add penalty terms to the loss function to prevent overfitting model parameters.  
3. **Reduce model complexity** – Use simpler models or limit parameters (e.g., prune decision trees, reduce neural network layers).  
4. **Collect more data / Data augmentation** – Increase training data volume or synthetically generate variations to improve robustness.
