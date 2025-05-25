# Road Accident Analysis Dashboard

A comprehensive dashboard for analyzing and visualizing road accident data, featuring predictive modeling and clustering analysis.

## Features

- Interactive data visualization dashboard
- Predictive modeling for accident analysis
- Clustering analysis for pattern recognition
- Data preparation and preprocessing
- Model diagnostics and evaluation

## Project Structure

```
.
├── dashboard.py              # Main Streamlit dashboard application
├── data_preparation.py       # Data preprocessing and cleaning
├── exploratory_analysis.py   # Exploratory data analysis
├── clustering_analysis.py    # Clustering algorithms and analysis
├── predictive_modeling.py    # Machine learning models
├── model_diagnostics.py      # Model evaluation and diagnostics
├── test_prediction.py        # Testing and prediction utilities
├── requirements.txt          # Project dependencies
└── road_accident_data_by_vehicle_type.csv  # Dataset
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd road
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the dashboard locally:

```bash
streamlit run dashboard.py
```

The dashboard will be available at:
- Local URL: http://localhost:8502
- Network URL: http://<your-ip>:8502

## Deployment

The application can be deployed on various platforms:

1. Streamlit Cloud (Recommended)
   - Push your code to GitHub
   - Connect your repository to Streamlit Cloud
   - Deploy directly from the Streamlit Cloud interface

2. Heroku
   - Follow the deployment instructions in the deployment guide
   - Ensure all dependencies are properly specified in requirements.txt

## Dependencies

- streamlit >= 1.24.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- plotly >= 5.13.0
- scikit-learn >= 1.2.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 