# Business Goal: 
## Answer the following question:

**“What are the most important factors influencing median house prices of neighborhoods of California, and implicitly, where should one look if they want to find affordable housing there?”**

To achieve the goal of the project, I designed and implemented an automated, cloud-based data pipeline that transforms raw housing data into a reliable, extended dataset ready for analysis and decision-making. The pipeline eliminates manual data preparation by systematically cleaning the data, extending it with external weather information, and making it easily accessible through SQL-based analytics tools.

By combining housing characteristics with contextual weather data, the project enables deeper insights into housing affordability, regional price differences, and location desirability. At the same time, the architecture is designed to be low-cost, scalable, and reproducible, ensuring the dataset can be refreshed regularly and extended in the future for more advanced analytics or predictive modeling.


---

## Stakeholders & Value

- **Data analysts / BI users**  
  Access to a clean and diverse dataset that can be queried with SQL for KPI reporting and exploratory analysis.

- **Data scientists**  
  Can experiment with models and feature relationships using a prepared dataset without spending time on data collection and cleaning.

- **Data engineering teams**  
  Having a fully serverless, automated pipeline that is easy to maintain, cost-efficient, and extendable with new data sources.

- **Business and real-estate decision makers, potential homebuyers**  
  Benefit from insights into regional price differences, housing affordability, and environmental context that support pricing and investment decisions.

---

## Designed Data Pipeline & Architecture

1. The raw California housing dataset is uploaded to and **Amazon S3 bucket**.  
2. An **EventBridge** schedule automatically triggers an **AWS Lambda** function to run the ETL pipeline.  
3. The Lambda function cleans the housing data, handles missing values, caps outliers, and creates a stratified sample to control external API usage.  
4. Lambda extends the housing data by calling an external weather API and aggregating temperature, sunshine duration, and precipitation data.  
5. The extended dataset is written back to **Amazon S3** as the processed output.  
6. An **AWS Glue crawler** scans the output bucket and registers the dataset in the **Glue Data Catalog**.  
7. **Amazon Athena** queries the cataloged table using SQL, making it available for analysis.
8. The transformed dataset is also used by a **Streamlit application**, which creates interactive visualizations (maps, distributions, scatter plots) and a machine-learning–based (Gradient Boosting) house price prediction.

---

## KPIs Supported by the Pipeline

- Median and average house value by region and ocean proximity  
- Price-to-income ratio measuring housing affordability  
- Price dispersion and volatility across different location categories  
- Relationships between environmental factors (sunshine, precipitation, temperature) and house prices  
- Housing Affordability Index: Extension of the price-to-income ratio: shows areas where housing is over- or under-valued      relative to local income level
- Model Prediction Accuracy: shows how well the model explains housing prices

---

## Costs and How the Architecture Supports the Goal

The pipeline relies on **serverless AWS services**, keeping infrastructure and operational costs low.

- **Amazon S3** provides cheap, durable storage for both raw and processed data.  
- **AWS Lambda** and **EventBridge** pay-per-execution, no idle compute costs.  
- **AWS Glue crawler** small, periodic costs while eliminating manual schema management.  
- **Amazon Athena** pay-per-query model, and the relatively small dataset size keeps query costs minimal.
- **Streamlit** runs locally, no cost.


This pipeline costs basically nothing, only 1-2 dollars a month. If I wanted to put this into serious commercial operation, the cost of the OpenMeteo-API would be 30 EUR for 1 million monthly API calls which would be more than enough.