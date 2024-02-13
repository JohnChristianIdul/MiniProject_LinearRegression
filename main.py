import pandas as pd

from linear_regression_model import *
from plotting_module import *
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(training_df, test_size=0.2, random_state=42)

# Specify the feature and the label.
# features = [col for col in train_df.columns if col != 'PremiumPrice']
features = [col for col in training_df.columns if col != 'PremiumPrice']

label = 'PremiumPrice'

# Step 3: Build the model
learning_rate = 5
model = build_model(learning_rate, features)

# Step 4: Train the model
epochs = 145
batch_size = 13
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


print("The learned bias for your model is %.4f\n" % trained_bias)

