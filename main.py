from data.synthetic_data import generate_data
from quantum_model.train_model import train_model
from quantum_model.evaluate_model import evaluate_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    params, cost_history = train_model(X_train, y_train)
    accuracy = evaluate_model(params, X_test, y_test)
    print(f'Test Accuracy: {accuracy}')
