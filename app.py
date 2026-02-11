import gradio as gr
import pickle
import numpy as np

# Load the trained model
with open("decision_tree_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict(feature1, feature2, feature3, feature4):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    return f"Prediction: {prediction[0]}"

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Feature 1"),
        gr.Number(label="Feature 2"),
        gr.Number(label="Feature 3"),
        gr.Number(label="Feature 4"),
    ],
    outputs="text",
    title="Decision Tree Prediction App",
    description="Enter the feature values to get prediction from the trained Decision Tree model."
)

if __name__ == "__main__":
    interface.launch()
