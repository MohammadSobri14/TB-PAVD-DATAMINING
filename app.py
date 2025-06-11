from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Load the Apriori model and dataset
with open('apriori_model.pkl', 'rb') as model_file:
    apriori_model = pickle.load(model_file)

# Assuming you also have a dataset for referencing products
# with open('product_data.pkl', 'rb') as data_file:
#     product_data = pickle.load(data_file)

# Initialize Flask app
app = Flask(__name__)

# Home route with form for product selection
@app.route('/')
def home():
    products = product_data['product_name'].unique()  # Assuming 'product_name' is the column in your dataset
    return render_template('index.html', products=products)

# Route to process recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_products = [
        request.form.get('product1'),
        request.form.get('product2'),
        request.form.get('product3'),
        request.form.get('product4'),
        request.form.get('product5')
    ]
    
    selected_products = [product for product in selected_products if product]  # Filter out None values

    # Check if any products are selected
    if not selected_products:
        return jsonify({"error": "No products selected"}), 400

    # Create a DataFrame for the selected products
    input_data = pd.DataFrame([selected_products], columns=[f'product_{i+1}' for i in range(len(selected_products))])

    # Generate recommendations using the Apriori model
    recommendations = apriori_model.recommend(input_data)  # Assuming your model has a recommend method

    return jsonify({"selected_products": selected_products, "recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
