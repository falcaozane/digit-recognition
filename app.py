import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Load the trained model
model = tf.keras.models.load_model("",compile=False)  

# Model predictions
y_pred_probabilities = model.predict(x_test)
y_pred_classes = y_pred_probabilities.argmax(axis=-1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Accuracy
acc_cm = accuracy_score(y_test, y_pred_classes)

# Define a Streamlit app
def main():
    st.title("MNIST Digit Recognition Web App")
    st.sidebar.header("Choose an option")

    # Option 1: Display an example image
    if st.sidebar.checkbox("Show Example Image"):
        st.image(x_test[0].squeeze(), caption="Example Image (Label: {})".format(y_test[0]), use_column_width=True)

    # Option 2: Display confusion matrix
    if st.sidebar.checkbox("Show Confusion Matrix"):
        st.subheader("Confusion Matrix")
        st.text("Accuracy: {:.2f}%".format(acc_cm * 100))
        normalized_cm = (cm / cm.max()) * 255
        st.image(normalized_cm, caption="Confusion Matrix", use_column_width=True, clamp=True)

        st.write("Confusion Matrix (Numbers):")
        st.text(cm)

    # Option 3: Display learning curve
    if st.sidebar.checkbox("Show Learning Curve"):
        st.subheader("Learning Curve")
        learning_curve()

    # Option 4: Upload an image for prediction
    if st.sidebar.checkbox("Upload Handwritten Digit Image"):
        st.subheader("Upload Handwritten Digit Image")
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
            image = np.array(image.resize((28, 28)))  # Resize to 28x28
            image = image / 255.0  # Normalize pixel values
            image = image.reshape(1, 28, 28, 1)  # Reshape for model input

            # Make prediction
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)

            st.subheader("Prediction:")
            st.image(image.squeeze(), caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### Predicted Digit: **{predicted_class}**")

# Learning curve function
def learning_curve():
    # Create some dummy data for the learning curve plot
    epoch_range = range(1, 11)
    training_accuracy = np.random.rand(10)
    validation_accuracy = np.random.rand(10)
    training_loss = np.random.rand(10)
    validation_loss = np.random.rand(10)

    # Training vs validation accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_range, training_accuracy)
    ax1.plot(epoch_range, validation_accuracy)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot(fig1)

    # Training vs validation loss
    fig2, ax2 = plt.subplots()
    ax2.plot(epoch_range, training_loss)
    ax2.plot(epoch_range, validation_loss)
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
