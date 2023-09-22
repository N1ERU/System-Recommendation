import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Step 1: Dataset Preparation
data = pd.read_csv("D:\Kuliah\RM\IndonesianUniversityRanking.csv")  # Replace "path_to_dataset.csv" with the actual path to your dataset file

# Perform data preprocessing steps as needed
# ...

# Step 2: Feature Selection
selected_features = ["Rank", "University", "Town"]  # Example features for university recommendation

X = data[selected_features]
y = data["Rank"]  # Replace "target_variable" with the column name of the target variable

# Step 3: Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Implementing the KNN Algorithm
numeric_features = ["Rank", "University", "Town"]  # Numeric features
categorical_features = []  # Categorical features

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

knn = KNeighborsClassifier(n_neighbors=5)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("knn", knn)
    ]
)

pipeline.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Hyperparameter Tuning
# Experiment with different values of k and evaluate the model's performance
# For example, you can use a loop to iterate over different values of k and select the best one based on the evaluation metric

best_k = None
best_accuracy = 0.0

for k in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("knn", knn)
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("Best k:", best_k)
print("Best Accuracy:", best_accuracy)

# Step 7: Deploying the Recommendation System
def get_user_preferences():
    # Implement a function to collect user preferences
    # Ask the user for their preferences on university rank, location, facilities, programs, etc.
    # Return the user preferences as a dictionary or a list
    
    preferences = {}
    preferences["Rank"] = input("Enter your preferred university rank: ")
    preferences["University"] = input("Enter your preferred location: ")
    preferences["Town"] = input("Enter your preferred facilities: ")
    
    return preferences

def generate_recommendations(preferences, knn_model):
    # Use the trained KNN model to generate personalized recommendations based on user preferences
    # Preprocess the user preferences similar to the training data
    
    # Convert the user preferences to a DataFrame
    user_preferences_df = pd.DataFrame(preferences, index=[0])
    
    # Scale the user preferences using the same scaler used on the training data
    user_preferences_scaled = preprocessor.transform(user_preferences_df)
    
    # Generate recommendations using the KNN model
    recommendations = knn_model.predict(user_preferences_scaled)
    
    return recommendations

user_preferences = get_user_preferences()
recommended_universities = generate_recommendations(user_preferences, pipeline)

print("Recommended universities:", recommended_universities)
