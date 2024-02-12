from linear_regression_model import *
from plotting_module import *
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(training_df, test_size=0.2, random_state=42)
print(training_df.columns)
# Specify the feature and the label.
features = ['Age', 'Height', 'Weight', 'PremiumPrice', 'Diabetes_0', 'Diabetes_1',
       'BloodPressureProblems_0', 'BloodPressureProblems_1',
       'AnyTransplants_0', 'AnyTransplants_1', 'AnyChronicDiseases_0',
       'AnyChronicDiseases_1', 'KnownAllergies_0', 'KnownAllergies_1',
       'HistoryOfCancerInFamily_0', 'HistoryOfCancerInFamily_1',
       'NumberOfMajorSurgeries_0', 'NumberOfMajorSurgeries_1',
       'NumberOfMajorSurgeries_2', 'NumberOfMajorSurgeries_3']
label = 'PremiumPrice'

# Step 3: Build the model
learning_rate = 0.01
model = build_model(learning_rate, features)

# Step 4: Train the model
epochs = 80
batch_size = 15
trained_weight, trained_bias, epochs, rmse = train_model(model, train_df, features, label, epochs, batch_size)

# Step 5: Plot the model's predictions against the actual labels
predictions = model.predict(test_df[features]).flatten()
plot_the_model(predictions, test_df[label])

# Step 6: Plot the loss curve
plot_the_loss_curve(epochs, rmse)

# Step 7: Print predicted and actual premium prices side by side
print("Sample   Predicted   Actual")
for i, (prediction, actual) in enumerate(zip(predictions, test_df[label])):
    print(f"{i+1:6}   ${prediction:.2f}    ${actual:.2f}")
