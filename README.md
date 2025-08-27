Segment AI: RFM Customer Segmentation System

🌟 Project Overview

Segment AI is a sophisticated customer segmentation platform that empowers businesses to move beyond a "one-size-fits-all" approach. By leveraging advanced machine learning, we transform raw transactional data into precise, actionable customer groups. This system provides a holistic view of customer value and behavior, enabling targeted marketing strategies that boost retention, optimize marketing spend, and maximize Customer Lifetime Value (CLTV).

This project was developed as a key deliverable for course "Computational Intelligence".The dataset used in this study is the "Customer Personality Analysis" dataset obtained from
kaggle (https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci). This Online Retail II data
set contains all the transactions occurring for a UK-based and registered, non-store online retail
between 01/12/2009 and 09/12/2011.


✨ Key Features:

Intelligent Customer Grouping: Utilizes Gaussian Mixture Models (GMM) to discover distinct customer segments based on spending habits. Unlike traditional methods, GMM handles complex, overlapping customer behaviors to provide a more accurate and realistic view.

Intuitive Auto-Naming System: Our custom rule-based system dynamically assigns business-friendly names to each segment, such as "Champions," "At-Risk," and "New Customers." This translates complex data outputs into clear, actionable insights for non-technical users.

Dynamic Visualizations: An interactive Treemap provides an instant, at-a-glance overview of segment sizes. Hovering over each segment reveals key RFM metrics, making data exploration intuitive and engaging.

End-to-End Automation: The system automates the entire process, from raw data ingestion and cleaning to RFM calculation, GMM clustering, and report generation, all through a simple web interface.

Effortless Reporting: Generates a comprehensive, multi-sheet Excel report that can be downloaded instantly. This report is ready for immediate use by marketing and sales teams for targeted campaigns.


🚀 The Tech Stack:

Frontend: HTML, CSS, JavaScript (Plotly.js for interactive visualizations)

Backend: Python (Flask for the web server)

Data Science Libraries:

Pandas & NumPy: For robust data manipulation and numerical operations.

Scikit-learn: Powers the core machine learning models, including GaussianMixture and StandardScaler.

Plotly & Matplotlib: For creating compelling and interactive data visualizations.

openpyxl


🧠 How It Works: A Look Inside:

Data Ingestion: A user uploads a transactional data file (e.g., online_retail_II.xlsx) through the web interface.

Preprocessing: The Flask backend automatically cleans the data, handling missing values, filtering irrelevant entries, and preparing it for analysis.

RFM Calculation: A custom engine computes Recency, Frequency, and Monetary values for each customer.

Feature Scaling: RFM values are standardized using StandardScaler to ensure each feature contributes equally to the clustering process.

GMM Clustering: The core GMM algorithm segments customers. The system first determines the optimal number of clusters by analyzing BIC/AIC scores, then applies the model.

Auto-Naming & Visualization: The system applies a dynamic rule set to name the clusters and generates the Treemap and detailed cluster profile table for immediate display.

Report Generation: An optimized report is created on the fly, containing all customer-level details, ready for download.

📁 File Structure

The project is organized in a clear, modular structure to separate the web application, data, and machine learning components.

/SegmentAI/

├── app.py  # Main Flask application with API endpoints

├── data/

│   └── online_retail_II.xlsx    # Dataset used for the project

├── templates/

│   └── index.html               # Frontend HTML for the web interface and Stylesheets for the frontent

├── static/

│   └── js/

│       └── script.js            # Frontend JavaScript for interactivity 

├── requirements.txt           # List of Python dependencies

└── README.md                  # This document

⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

Prerequisites

Python 3.8 or higher

Git (optional, but recommended for cloning)

Step 1: Clone the Repository

First, clone the project repository from GitHub to your local machine.

git clone https://github.com/your-username/your-repository.git
cd your-repository

Step 2: Create and Activate Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

# On macOS and Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate



Step 3: Install Dependencies
Install all required Python libraries using the requirements.txt file.

pip install -r requirements.txt

Step 4: Run the Application
Once the dependencies are installed, you can start the Flask application.

python app.py

The application will now be running on http://127.0.0.1:5000. Open this URL in your web browser to interact with the system.

🤝 Let's Connect
This project demonstrates a blend of machine learning expertise and a strong focus on building user-friendly, impactful applications. I'm actively seeking opportunities where I can apply my skills in data science, machine learning, and full-stack development.

LinkedIn: https://www.linkedin.com/in/rashadul-nafis-riyad/

Email: nafisrashadul@gmail.com

