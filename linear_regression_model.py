import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
training_df = pd.read_csv("medical_premium.csv")

# Specify the columns to exclude from one-hot encoding (usually continuous variables and the target variable)
exclude_columns = ['PremiumPrice']

# One-hot encode categorical variables (excluding 'PremiumPrice', and optionally excluding 'Age' and 'Weight'
# if they're continuous)
training_df_encoded = pd.get_dummies(training_df, columns=[col for col in training_df.columns if col not in exclude_columns])

# Convert dataframe to float type
training_df = training_df.astype(float)


# Define the model
def build_model(my_learning_rate, my_features):
    """Create and compile a multiple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a multiple linear regression model
    # is a single node in a single layer with the number of inputs equal to the number of features.
    model.add(tf.keras.layers.Dense(units=1,
                                    input_shape=(len(my_features),)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, df, features, label, epochs, batch_size):
    """Train the model by feeding it data."""
    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[features], y=df[label], batch_size=batch_size, epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse


print("Defined the build_model and train_model functions.")