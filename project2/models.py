import torch
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def baseline():
    # Baseline model that predicts the most frequent class
    baseline_model = DummyClassifier(strategy="most_frequent")
    return baseline_model


def logistic_regression(lam):
    # Logistic Regression model with multiclass support and regularization parameter
    logistic_model = LogisticRegression(
        solver="newton-cg",
        multi_class="multinomial",
        tol=1e-4,
        random_state=1,
        penalty="l2",
        C=1 / lam,
    )
    return logistic_model

def ann(h):
    # Define the number of input features (from the dataset), hidden units, and classes
    n_features = 9  # Assuming 9 features in the Glass dataset
    n_hidden_units = h  # Adjust this based on your chosen complexity-controlling parameter
    n_classes = 7  # Assuming 7 classes in the Glass dataset

    # Define the ANN model using a lambda function with torch's Sequential API
    ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, n_classes)  # Output layer for multiclass classification
    )
    return ann_model

def train_neural_net(model, loss_fn, X, y, n_replicates=3, max_iter=100, patience=100):
    """Train a neural network with PyTorch, including early stopping and multiple replicates."""
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    best_final_loss = float('inf')
    final_net = None

    for replicate in range(n_replicates):
        print(f'\n\tReplicate {replicate + 1}/{n_replicates}')
        
        net = model()
        optimizer = torch.optim.Adam(net.parameters())
        
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(max_iter):
            net.train()
            y_est = net(X)
            loss = loss_fn(y_est, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} with loss {best_loss}")
                break
        
        # Check if this replicate's model is the best so far
        if best_loss < best_final_loss:
            best_final_loss = best_loss
            final_net = net
    
    return final_net, best_final_loss


