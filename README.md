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
