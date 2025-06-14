# Wine Quality Analysis using Support Vector Machine (SVM)

A comprehensive machine learning project that implements Support Vector Machine algorithms to classify wine quality based on chemical properties. This binary classification system distinguishes between good and poor quality wines using advanced data preprocessing and model evaluation techniques.

## 🎯 Project Overview

This Artificial Intelligence project leverages Support Vector Machine (SVM) algorithms to predict wine quality classification. The system analyzes various chemical features of wines and uses binary classification to determine quality levels, providing valuable insights for wine quality assessment.

### Key Objectives

- Implement robust wine quality classification using SVM
- Analyze the relationship between chemical properties and wine quality
- Provide comprehensive model evaluation and performance metrics
- Deliver actionable insights through feature importance analysis

## ✨ Features

### Data Processing

- **Comprehensive Data Loading**: Automated loading and inspection of wine quality datasets
- **Exploratory Data Analysis**: Statistical analysis and data distribution visualization
- **Advanced Preprocessing**: Data normalization using StandardScaler for optimal model performance
- **Quality Assurance**: Data validation and cleaning procedures

### Machine Learning Implementation

- **Dual Kernel SVM**: Implementation of both RBF and Linear kernel SVM models
- **Binary Classification**: Efficient good/bad quality wine classification
- **Cross-Validation**: Robust model validation using train-validation split methodology
- **Hyperparameter Optimization**: Fine-tuned parameters for optimal performance

### Model Evaluation & Analysis

- **Comprehensive Metrics**: Accuracy, precision, recall, and F1-score calculations
- **Confusion Matrix**: Visual representation of model predictions and errors
- **Feature Importance Analysis**: Identification of most influential chemical properties
- **Performance Visualization**: Professional charts and graphs for result interpretation

### Visualization Suite

- **Confusion Matrix Heatmaps**: Clear visualization of classification results
- **Feature Importance Charts**: Ranking of chemical properties by predictive power
- **Data Distribution Plots**: Understanding of dataset characteristics
- **Performance Metrics Dashboard**: Comprehensive model evaluation displays

## 🛠️ Technical Requirements

### System Requirements

- **Python Version**: 3.7 or higher (Python 3.8+ recommended)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for larger datasets)
- **Storage**: At least 100MB free space

### Virtual Environment Setup

Using a virtual environment is **strongly recommended** to avoid dependency conflicts:

#### Linux/macOS Setup

```bash
# Create virtual environment
python3 -m venv wine_analysis_env

# Activate virtual environment
source wine_analysis_env/bin/activate

# Verify activation
which python
```

#### Windows Setup

```bash
# Create virtual environment
python -m venv wine_analysis_env

# Activate virtual environment
wine_analysis_env\Scripts\activate

# Verify activation
where python
```

#### Deactivation (All Platforms)

```bash
deactivate
```

### Dependencies Installation

#### Option 1: Individual Package Installation

```bash
pip install pandas==2.1.4
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
```

#### Option 2: All Dependencies at Once

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

#### Option 3: Requirements File (Recommended)

Create a `requirements.txt` file:

```txt
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Package Details

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and tools
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization

## 📊 Dataset Requirements

### File Structure

The project requires two CSV files in the same directory as the main script:

- `wineQuality_train.data` - Training dataset for model learning
- `wineQuality_val.data` - Validation dataset for model evaluation

### Dataset Format

Both files must contain:

#### Feature Columns (Chemical Properties)

- **Acidity levels**: Fixed acidity, volatile acidity, citric acid
- **Sugar content**: Residual sugar
- **Chemical compounds**: Chlorides, sulfur dioxide levels
- **Physical properties**: Density, pH, alcohol content
- **Additional features**: Any other relevant chemical measurements

#### Target Column

- **quality**: Binary classification column
  - `0` = Poor/Bad quality wine
  - `1` = Good/High quality wine

### Data Quality Requirements

- No missing values in critical columns
- Consistent data types across features
- Balanced or documented class distribution
- Proper numerical encoding for all features

## 🚀 Getting Started

### Step-by-Step Installation Guide

1. **Download the Project**

   ```bash
   git clone <repository-url>
   cd wine-quality-analysis
   ```

2. **Set Up Virtual Environment**

   ```bash
   python -m venv wine_analysis_env
   source wine_analysis_env/bin/activate  # Linux/macOS
   # OR
   wine_analysis_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**
   - Place `wineQuality_train.data` in the project directory
   - Place `wineQuality_val.data` in the project directory
   - Verify file formats and column names

5. **Run the Analysis**

   ```bash
   python wine_quality.py
   ```

### Quick Start (Advanced Users)

```bash
# One-line setup for experienced users
python -m venv env && source env/bin/activate && pip install pandas numpy scikit-learn matplotlib seaborn && python wine_quality.py
```

## 📁 Project Structure

```txt
wine-quality-analysis/
├── 📄 wine_quality.py              # Main analysis script
├── 📊 wineQuality_train.data       # Training dataset (user provided)
├── 📊 wineQuality_val.data         # Validation dataset (user provided)
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # Project documentation (this file)
├── 📁 results/                     # Generated outputs (optional)
│   ├── 🖼️ confusion_matrix.png     # Confusion matrix visualization
│   ├── 📊 feature_importance.png   # Feature importance chart
│   └── 📈 model_performance.txt    # Performance metrics report
└── 📁 docs/                       # Additional documentation (optional)
    ├── 📄 methodology.md           # Technical methodology
    └── 📄 interpretation_guide.md  # Results interpretation guide
```

## 📈 Expected Output

### Console Output

The script provides comprehensive console output including:

1. **Dataset Information**
   - Training set size and feature count
   - Validation set size and class distribution
   - Data quality assessment results

2. **Model Training Progress**
   - Training time and convergence information
   - Cross-validation scores
   - Hyperparameter optimization results

3. **Performance Metrics**
   - **Accuracy**: Overall classification accuracy percentage
   - **Precision**: True positive rate for quality predictions
   - **Recall**: Sensitivity of the model to actual quality wines
   - **F1-Score**: Harmonic mean of precision and recall

4. **Feature Analysis**
   - Ranked list of most important chemical properties
   - Feature importance scores and interpretations
   - Recommendations for quality improvement

### Visual Outputs

- **Confusion Matrix Heatmap**: Color-coded prediction accuracy visualization
- **Feature Importance Bar Chart**: Ranked visualization of chemical property importance
- **ROC Curve**: Model performance across different thresholds
- **Data Distribution Plots**: Understanding of input data characteristics

## 🤖 Model Architecture

### Algorithm Details

- **Primary Algorithm**: Support Vector Machine (SVM)
- **Kernel Functions**:
  - RBF (Radial Basis Function) - Primary classification
  - Linear - Feature importance analysis
- **Classification Type**: Binary classification (Good vs. Poor quality)

### Preprocessing Pipeline

1. **Data Loading**: Automated CSV file reading with error handling
2. **Feature Selection**: Automatic detection of feature columns
3. **Data Normalization**: Z-score standardization using StandardScaler
4. **Train-Validation Split**: Proper dataset separation for unbiased evaluation

### Model Training Process

1. **Data Preprocessing**: Normalization and feature engineering
2. **Model Initialization**: SVM with optimized hyperparameters
3. **Training Phase**: Model learning on training dataset
4. **Validation Phase**: Performance evaluation on unseen data
5. **Feature Analysis**: Linear SVM coefficient analysis for interpretability

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### File Not Found Errors

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solutions**:

- Verify dataset files are named exactly `wineQuality_train.data` and `wineQuality_val.data`
- Ensure files are in the same directory as `wine_quality.py`
- Check file permissions and accessibility
- Verify file format (CSV with proper encoding)

#### Import/Dependency Errors

**Problem**: `ModuleNotFoundError: No module named 'package_name'`
**Solutions**:

```bash
# Reinstall specific package
pip install --upgrade package_name

# Reinstall all dependencies
pip install --force-reinstall -r requirements.txt

# Check Python and pip versions
python --version
pip --version
```

#### Memory Issues

**Problem**: `MemoryError` or system slowdown
**Solutions**:

- Reduce dataset size for testing: `df.sample(frac=0.1)`
- Use a machine with more RAM (8GB+ recommended)
- Implement batch processing for very large datasets
- Consider using sparse matrices for large feature sets

#### Data Format Issues

**Problem**: `ValueError: could not convert string to float`
**Solutions**:

- Check for non-numeric data in feature columns
- Verify column names match expected format
- Handle missing values appropriately
- Ensure consistent data encoding

#### Performance Issues

**Problem**: Slow execution or poor model performance
**Solutions**:

- Verify data quality and class balance
- Adjust SVM hyperparameters (C, gamma)
- Consider feature selection or dimensionality reduction
- Check for data leakage or overfitting

### Getting Help

If you encounter issues not covered here:

1. Check the console output for specific error messages
2. Verify all requirements are met
3. Test with a smaller dataset first
4. Consider reaching out to the project maintainer

## 👨‍💻 Author & Credits

**Michel Ferreira Viana de Carvalho**  
*Artificial Intelligence Project (EER)*

### Acknowledgments

- Built using scikit-learn machine learning library
- Visualization powered by matplotlib and seaborn
- Data processing with pandas and numpy

## 📄 License

This project is part of an academic assignment. Please respect academic integrity guidelines when using or referencing this work.

## 🔄 Version History

- **v1.0.0**: Initial release with basic SVM implementation
- **v1.1.0**: Added feature importance analysis and improved visualizations
- **v1.2.0**: Enhanced error handling and documentation

## 🚀 Future Enhancements

- [ ] Multi-class classification for detailed quality levels
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation implementation
- [ ] Feature selection algorithms
- [ ] Model comparison with other algorithms (Random Forest, XGBoost)
- [ ] Web interface for interactive predictions
- [ ] API endpoint for real-time classification

---

**Note**: This project is designed for educational and research purposes. Ensure you have proper rights to use any wine quality datasets before implementation.
