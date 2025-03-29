# Real_Estate_Price_Prediction_application
This app has been built using Streamlit and deployed with Streamlit community cloud

This application predicts whether someone is eligible for a loan based on inputs derived from the German Credit Risk dataset. The model aims to help users assess loan eligibility by leveraging machine learning predictions.
Features
•	User-friendly interface powered by Streamlit.
•	Input form to enter details such as credit history, loan amount, income, and other relevant factors.
•	Real-time prediction of loan eligibility based on the trained model.
•	Accessible via Streamlit Community Cloud.
Dataset
The application is trained on the given dataset. It includes features like:
•	Year_sold
•	Property_tax
•	Insurance
•	Beds
•	Baths
•	Sqft
•	Year_built
•	Lot-size
•	Basement
•	Property_type

Technologies Used
•	Streamlit: For building the web application.
•	Scikit-learn: For model training and evaluation.
•	Pandas and NumPy: For data preprocessing and manipulation.
•	Matplotlib and Seaborn: For exploratory data analysis and visualization (if applicable).

Model
The predictive model is trained using the Linear Regression Model. It applies preprocessing steps like encoding categorical variables and scaling numerical features. 

Installation (for local deployment)
If you want to run the application locally, follow these steps:
1.	Clone the repository:
2.	git clone https://github.com/chen041081733/2216-project-part1.git
3.	cd 2216-project-part1
4.	Create and activate a virtual environment:
python -m venv env
5.	source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
6.	Install dependencies:
pip install -r requirements.txt
7.	Run the Streamlit application:
streamlit run part1_streamlit.py

Thank you for using the Real Estate Price Prediction App! Feel free to share your feedback.

