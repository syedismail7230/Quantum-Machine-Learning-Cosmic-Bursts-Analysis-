import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data.synthetic_data import generate_data
from quantum_model.train_model import train_model
from quantum_model.evaluate_model import evaluate_model
from quantum_model.quantum_circuit import create_quantum_circuit

# Generate and display data
st.title('Quantum Machine Learning for Cosmic Burst Analysis')
st.write('This app demonstrates quantum machine learning to analyze cosmic bursts using synthetic data.')

st.header('1. Generate Data')
n_samples = st.sidebar.slider('Number of Samples', min_value=5, max_value=5000, value=1000, key='samples_slider')
n_features = st.sidebar.slider('Number of Features', min_value=2, max_value=10, value=5, key='features_slider')
noise = st.sidebar.slider('Noise Level', min_value=0.0, max_value=1.0, value=0.1, key='noise_slider')

# Generate data and include impact
try:
    X_train, X_test, y_train, y_test, impact_train, impact_test = generate_data(
        n_samples=n_samples, n_features=n_features, noise=noise
    )
except ValueError as e:
    st.error(f"Data generation error: {e}")
else:
    st.write(f'Generated {n_samples} synthetic data points with {n_features} features each.')

    # Display sample data
    st.write('Training data (first 5 samples):')
    st.write(X_train[:5])
    st.write('Test data (first 5 samples):')
    st.write(X_test[:5])

    # Train the model
    st.header('2. Train Quantum Machine Learning Model')
    try:
        params, cost_history = train_model(X_train, y_train)
    except Exception as e:
        st.error(f"Training error: {e}")
    else:
        st.write('Training completed.')

        # Plot training cost
        st.subheader('Training Cost Over Epochs')
        fig, ax = plt.subplots()
        if cost_history:
            ax.plot(cost_history)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Cost')
            ax.set_title('Training Cost Over Epochs')
            st.pyplot(fig)
        else:
            st.warning("Cost history is empty. Check the training process.")

        # Evaluate the model
        st.header('3. Evaluate Model')
        try:
            accuracy = evaluate_model(params, X_test, y_test)
        except Exception as e:
            st.error(f"Evaluation error: {e}")
        else:
            st.write(f'Test Accuracy: {accuracy:.2f}')

            # Display predictions
            st.subheader('Predictions')

            # Create the quantum circuit again with the correct number of qubits and layers
            n_qubits = X_test.shape[1]
            n_layers = params.shape[0]
            quantum_circuit = create_quantum_circuit(n_qubits, n_layers)

            try:
                predictions = [quantum_circuit(params, x) for x in X_test]
                predictions = np.array(predictions)
                predictions = (predictions[:, 0] > 0).astype(int)
                st.write('Predictions on the test set (first 5 samples):')
                st.write(predictions[:5])
            except Exception as e:
                st.error(f"Prediction error: {e}")

            # Visualize Qubit Data (if needed)
            st.header('Qubit Data Visualization')
            st.write('Visualize the first qubit data:')
            fig, ax = plt.subplots()
            ax.bar(range(len(X_test[0])), X_test[0])
            ax.set_xlabel('Qubit Index')
            ax.set_ylabel('Qubit Value')
            ax.set_title('Qubit Data for the First Test Sample')
            st.pyplot(fig)

            # Analyze the impact of cosmic bursts
            st.header('Impact Analysis of Cosmic Bursts')
            st.write('Analyze the impact of cosmic bursts after they occur.')

            # Example analysis: Correlate impact with features
            try:
                impact_analysis = np.corrcoef(X_train.T, impact_train)
                st.write('Correlation between features and impact:')
                st.write(impact_analysis)
            except Exception as e:
                st.error(f"Impact analysis error: {e}")

            # Visualize impact data
            st.subheader('Impact Visualization')
            fig, ax = plt.subplots()
            ax.scatter(range(len(impact_test)), impact_test, alpha=0.5)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Impact')
            ax.set_title('Impact of Cosmic Bursts (Test Data)')
            st.pyplot(fig)
