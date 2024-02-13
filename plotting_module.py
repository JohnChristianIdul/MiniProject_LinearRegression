from matplotlib import pyplot as plt


def plot_the_model(predictions, labels):
    """Plot the predicted charges against the actual price."""
    plt.figure(figsize=(8, 6))  # Optional: Adjusts the figure size for better readability
    plt.scatter(labels, predictions, alpha=0.5)  # alpha is set to make dots semi-transparent
    plt.xlabel("Actual charges")
    plt.ylabel("Predicted charges")
    plt.title("Actual vs. Predicted charges")

    # Add a red line that bisects the graph
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red')  # Line from min to max

    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""
    plt.plot(epochs, rmse, label="Root Mean Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Root Mean Squared Error vs. Epoch")
    plt.legend()
    plt.show()