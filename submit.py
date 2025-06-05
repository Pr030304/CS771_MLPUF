import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import time

# Global variables for model parameters
W = None
b = 0

def my_map(c):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    n = c.shape[0]
    feat = np.zeros((n, 100), dtype=np.float64)
    D = 1 - 2 * c  # Bipolar encoding: {0,1} → {+1,-1}

    # Compute x_i = ∏_{k=i}^7 d_k for i = 0 to 6
    x = np.ones((n, 7), dtype=np.float64)
    for i in range(7):
        x[:, i] = np.prod(D[:, i:], axis=1)

    # Compute features
    for idx in range(n):
        c_i = c[idx]
        x_i = x[idx]
        vec = []

        # (1) c_i * c_j, i < j: 28 terms
        for i in range(8):
            for j in range(i + 1, 8):
                vec.append(c_i[i] * c_i[j])

        # (2) c_i * x_j, exclude c_i x_{i+1} (i=0 to 5), c_6 x_6, c_7 x_6: 48 terms
        for i in range(8):
            for j in range(7):
                if not ((j == i + 1 and i <= 5) or (j == 6 and i in [6, 7])):
                    vec.append(c_i[i] * x_i[j])

        # (3) x_i * x_j, i < j, exclude x_i x_{i+1}, x_i x_{i+2}: 10 terms
        for i in range(7):
            for j in range(i + 1, 7):
                if j not in [i + 1, i + 2]:
                    vec.append(x_i[i] * x_i[j])

        # (4) x_i, i = 0 to 5: 6 terms
        vec.extend(x_i[:6])

        # (5) c_i, i = 0 to 7: 8 terms
        vec.extend(c_i)

        # (6) Constant term: 1 term
        #vec.append(1.0)

        feat[idx] = np.array(vec, dtype=np.float64)

    return feat

def my_fit(X_train, y_train):
    ################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    # X_train = train_data[:, :8]
    # y_train = train_data[:, 8].astype(int)

    # Map features with timing
    start_map_time = time.time()
    X_mapped = my_map(X_train)
    map_time = time.time() - start_map_time
    print(f"Feature mapping time for training data: {map_time:.4f} seconds")

    # Train LinearSVC with timing
    start_train_time = time.time()
    # model = LinearSVC(loss='squared_hinge', penalty='l2', tol=1e-3, C=0.01, max_iter=5000, dual=False)
    model = LogisticRegression(penalty='l2', C=100, tol =1e-3, solver='lbfgs', max_iter=5000)
    model.fit(X_mapped, y_train)
    train_time = time.time() - start_train_time

    # Compute training accuracy
    train_predictions = model.predict(X_mapped)
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Training time: {train_time:.4f} seconds")

    # Extract weights and bias
    global W, b
    W = model.coef_.flatten()
    b = model.intercept_[0]

    return W, b

# def my_predict(X_test, W, b):
#     # Map features with timing
#     start_map_time = time.time()
#     X_mapped = my_map(X_test)
#     map_time = time.time() - start_map_time
#     print(f"Feature mapping time for test data: {map_time:.4f} seconds")

#     predictions = np.sign(X_mapped @ W + b)
#     return (1 + predictions) / 2  # Convert to {0,1}

# if __name__ == "__main__":
#     # Load data
#     train_data = np.loadtxt("public_trn.txt")
#     test_data = np.loadtxt("public_tst.txt")
#     X_train = train_data[:, :8]
#     y_train = train_data[:, 8].astype(int)
#     X_test = test_data[:, :8]
#     y_test = test_data[:, 8].astype(int)

#     # Train and predict
#     W, b = my_fit(train_data)
#     predictions = my_predict(X_test, W, b)

#     # Compute and print accuracy
#     correct_predictions = np.sum(predictions == y_test)
#     total_predictions = len(y_test)
#     accuracy = correct_predictions / total_predictions
#     misclassification_rate = 1 - accuracy

#     # Detailed misclassification analysis
#     misclassified_indices = np.where(predictions != y_test)[0]
#     print(f"Number of correct predictions: {correct_predictions}/{total_predictions}")
#     print(f"Test Accuracy: {accuracy:.4f}")
#     print(f"Misclassification Rate: {misclassification_rate:.4f}")
#     print(f"Number of misclassified samples: {len(misclassified_indices)}")
#     if len(misclassified_indices) > 0:
#         print(f"Sample misclassified indices: {misclassified_indices[:5]}")
#         for idx in misclassified_indices[:5]:
#             print(f"Misclassified sample {idx}: Challenge = {X_test[idx]}, Predicted = {predictions[idx]}, Actual = {y_test[idx]}")
################################
# Non Editable Region Starting #
################################
################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################
    """
    Recover non-negative delays (p,q,r,s) for a 64-bit Arbiter PUF
    from a 65-dim linear model w (first 64 = weights, last = bias).
    Returns four (64,) arrays: p, q, r, s.
    """
    # 1) Split weights and bias
    W = w[:64]
    b = w[64]

    # 2) Reconstruct alpha, beta
    alpha = np.empty(64, dtype=float)
    beta  = np.empty(64, dtype=float)

    alpha[0] = W[0]
    diffs = np.diff(W)       # length 63
    beta[:-1] = diffs
    alpha[1:] = W[1:] - beta[:-1]
    beta[63] = b

    # 3) Solve for delays ensuring non-negativity
    A = alpha + beta  # = p - q
    B = alpha - beta  # = r - s

    p = np.maximum(A, 0.0)
    q = np.maximum(-A, 0.0)
    r = np.maximum(B, 0.0)
    s = np.maximum(-B, 0.0)

    return p, q, r, s