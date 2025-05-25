# 🚗 Road Accident Analysis Dashboard

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://road-accident-dashboard.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.9.18-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*A comprehensive dashboard for analyzing and visualizing road accident data, featuring predictive modeling and clustering analysis.*

[Live Demo](https://road-accident-dashboard.streamlit.app/) | [Documentation](#documentation) | [Installation](#installation) | [Features](#features)

</div>

## 📊 Live Demo

The dashboard is now live and can be accessed at: [Road Accident Analysis Dashboard](https://road-accident-dashboard.streamlit.app/)

<div align="center">
  <img src="https://img.shields.io/badge/Status-Live-success" alt="Status: Live"/>
  <img src="https://img.shields.io/badge/Platform-Streamlit Cloud-blue" alt="Platform: Streamlit Cloud"/>
</div>

## ✨ Features

- 📈 **Interactive Data Visualization Dashboard**
  - Real-time data exploration
  - Customizable charts and graphs
  - Interactive filtering and analysis

- 🤖 **Predictive Modeling**
  - Advanced machine learning algorithms
  - Accurate accident prediction
  - Model performance metrics

- 🔍 **Clustering Analysis**
  - Pattern recognition in accident data
  - Geographic clustering
  - Risk assessment visualization

- 📊 **Data Preparation & Preprocessing**
  - Automated data cleaning
  - Feature engineering
  - Data quality assurance

- 📈 **Model Diagnostics & Evaluation**
  - Comprehensive model validation
  - Performance metrics
  - Error analysis

## 📁 Project Structure

```
.
├── 📊 dashboard.py              # Main Streamlit dashboard application
├── 🔧 data_preparation.py       # Data preprocessing and cleaning
├── 📈 exploratory_analysis.py   # Exploratory data analysis
├── 🔍 clustering_analysis.py    # Clustering algorithms and analysis
├── 🤖 predictive_modeling.py    # Machine learning models
├── 📊 model_diagnostics.py      # Model evaluation and diagnostics
├── 🧪 test_prediction.py        # Testing and prediction utilities
├── 📋 requirements.txt          # Project dependencies
└── 📁 road_accident_data_by_vehicle_type.csv  # Dataset
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd road
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

```bash
streamlit run dashboard.py
```

The dashboard will be available at:
- 🌐 Local URL: http://localhost:8502
- 🌍 Network URL: http://<your-ip>:8502

## 🚀 Deployment

The application is currently deployed on Streamlit Cloud and can be accessed at:
- [https://road-accident-dashboard.streamlit.app/](https://road-accident-dashboard.streamlit.app/)

### Deployment Options

1. **Streamlit Cloud** (Recommended)
   - Push your code to GitHub
   - Connect your repository to Streamlit Cloud
   - Deploy directly from the Streamlit Cloud interface

2. **Heroku**
   - Follow the deployment instructions in the deployment guide
   - Ensure all dependencies are properly specified in requirements.txt

## 📦 Dependencies

<div align="center">

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.24.0 | Web application framework |
| pandas | ≥ 1.5.0 | Data manipulation |
| numpy | ≥ 1.23.0 | Numerical computing |
| plotly | ≥ 5.13.0 | Interactive visualizations |
| scikit-learn | ≥ 1.2.0 | Machine learning |
| scipy | ≥ 1.10.0 | Scientific computing |
| matplotlib | ≥ 3.7.0 | Static visualizations |
| seaborn | ≥ 0.12.0 | Statistical visualizations |

</div>

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by the Road Accident Analysis Team</sub>
</div> 
