import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)


class WineQualityAnalyzer:
    """
    A comprehensive wine quality analysis system using Support Vector Machine.

    This class handles data loading, preprocessing, model training, evaluation,
    and visualization for wine quality classification tasks.
    """

    def __init__(self):
        """Initialize the Wine Quality Analyzer."""
        self.train_df = None
        self.val_df = None
        self.features = None
        self.target_col = None
        self.scaler = StandardScaler()
        self.svm_model = None
        self.svm_linear = None
        self.results = {}

        print("=" * 60)
        print("🍷 WINE QUALITY ANALYSIS USING SVM")
        print("=" * 60)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Author: Michel Ferreira Viana de Carvalho")
        print("=" * 60)

    def load_data(self):
        """
        Load training and validation datasets from CSV files.

        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            print("\n📁 LOADING DATASETS")
            print("-" * 30)

            # Load real data files
            self.train_df = pd.read_csv("wineQuality_train.data")
            self.val_df = pd.read_csv("wineQuality_val.data")

            print("✅ Data loaded successfully from original files!")
            print(f"   Training set: {self.train_df.shape[0]} samples")
            print(f"   Validation set: {self.val_df.shape[0]} samples")
            print(f"   Total features: {self.train_df.shape[1] - 1}")

            return True

        except FileNotFoundError as e:
            print(f"❌ Error: File not found - {e}")
            print(
                "   Please ensure 'wineQuality_train.data' and 'wineQuality_val.data'"
            )
            print("   are in the current directory")
            return False

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def inspect_data(self):
        """Perform comprehensive data inspection and validation."""
        print("\n🔍 DATA INSPECTION")
        print("-" * 30)

        # Display basic information
        print(f"Training dataset shape: {self.train_df.shape}")
        print(f"Validation dataset shape: {self.val_df.shape}")

        # Display columns
        print(f"\nDataset columns: {list(self.train_df.columns)}")

        # Identify features and target automatically
        self.features = [
            col for col in self.train_df.columns if col.lower() != "quality"
        ]

        # Find target column
        if "quality" in self.train_df.columns:
            self.target_col = "quality"
        else:
            # Try to find similar column
            possible_targets = [
                col for col in self.train_df.columns if "quality" in col.lower()
            ]
            if possible_targets:
                self.target_col = possible_targets[0]
                print(f"⚠️  Using '{self.target_col}' as target column")
            else:
                print("❌ Error: 'quality' column not found in dataset")
                return False

        print(f"\nTarget column: {self.target_col}")
        print(f"Feature columns ({len(self.features)}): {self.features}")

        # Display first few rows
        print("\n📊 First 5 rows of training data:")
        print(self.train_df.head())

        # Display class distribution
        print("\n📈 Class Distribution:")
        train_dist = self.train_df[self.target_col].value_counts().sort_index()
        val_dist = self.val_df[self.target_col].value_counts().sort_index()

        for class_val in sorted(self.train_df[self.target_col].unique()):
            train_count = train_dist.get(class_val, 0)
            val_count = val_dist.get(class_val, 0)
            class_name = "Poor Quality" if class_val == 0 else "Good Quality"
            print(f"   Class {class_val} ({class_name}):")
            print(
                f"     Training: {train_count} samples ({train_count/len(self.train_df)*100:.1f}%)"
            )
            print(
                f"     Validation: {val_count} samples ({val_count/len(self.val_df)*100:.1f}%)"
            )

        # Display feature statistics
        print("\n📊 Feature Value Ranges:")
        for feature in self.features:
            min_val = self.train_df[feature].min()
            max_val = self.train_df[feature].max()
            mean_val = self.train_df[feature].mean()
            std_val = self.train_df[feature].std()
            print(
                f"   {feature}: [{min_val:.3f}, {max_val:.3f}] "
                f"(μ={mean_val:.3f}, σ={std_val:.3f})"
            )

        # Check for missing values
        missing_train = self.train_df.isnull().sum().sum()
        missing_val = self.val_df.isnull().sum().sum()

        if missing_train > 0 or missing_val > 0:
            print(f"\n⚠️  Missing values detected:")
            print(f"   Training set: {missing_train}")
            print(f"   Validation set: {missing_val}")
        else:
            print("\n✅ No missing values detected")

        return True

    def preprocess_data(self):
        """Preprocess data with normalization and feature scaling."""
        print("\n⚙️  DATA PREPROCESSING")
        print("-" * 30)

        # Separate features and targets
        X_train = self.train_df[self.features]
        y_train = self.train_df[self.target_col]
        X_val = self.val_df[self.features]
        y_val = self.val_df[self.target_col]

        print("Before normalization:")
        print(f"   Mean of training features: {X_train.mean().mean():.3f}")
        print(f"   Std of training features: {X_train.std().mean():.3f}")

        # Apply StandardScaler normalization (z-score)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print("\nAfter StandardScaler normalization:")
        print(f"   Mean of training features: {X_train_scaled.mean():.3f}")
        print(f"   Std of training features: {X_train_scaled.std():.3f}")

        # Store processed data
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_train = y_train
        self.y_val = y_val

        print("✅ Data preprocessing completed successfully")

        return True

    def train_models(self):
        """Train both RBF and Linear SVM models."""
        print("\n🤖 MODEL TRAINING")
        print("-" * 30)

        # Train RBF SVM (primary model)
        print("Training SVM with RBF kernel...")
        self.svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        self.svm_model.fit(self.X_train_scaled, self.y_train)

        print(f"✅ RBF SVM trained successfully!")
        print(f"   Kernel: {self.svm_model.kernel}")
        print(f"   C parameter: {self.svm_model.C}")
        print(f"   Gamma: {self.svm_model.gamma}")
        print(f"   Support vectors: {self.svm_model.n_support_}")

        # Train Linear SVM (for feature importance analysis)
        print("\nTraining SVM with Linear kernel for feature analysis...")
        self.svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
        self.svm_linear.fit(self.X_train_scaled, self.y_train)

        print("✅ Linear SVM trained successfully!")

        return True

    def evaluate_models(self):
        """Evaluate both models and generate comprehensive metrics."""
        print("\n📊 MODEL EVALUATION")
        print("-" * 30)

        # RBF SVM Evaluation
        y_pred_rbf = self.svm_model.predict(self.X_val_scaled)

        # Linear SVM Evaluation
        y_pred_linear = self.svm_linear.predict(self.X_val_scaled)

        # Calculate metrics for both models
        self.results = {
            "rbf": self._calculate_metrics(self.y_val, y_pred_rbf, "RBF SVM"),
            "linear": self._calculate_metrics(self.y_val, y_pred_linear, "Linear SVM"),
        }

        # Display comparison
        print(f"\n📈 MODEL COMPARISON")
        print("-" * 30)
        print(f"{'Metric':<12} {'RBF SVM':<10} {'Linear SVM':<10}")
        print("-" * 32)

        metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in metrics:
            rbf_val = self.results["rbf"][metric]
            linear_val = self.results["linear"][metric]
            print(f"{metric.capitalize():<12} {rbf_val:<10.4f} {linear_val:<10.4f}")

        # Store best predictions (using RBF as primary)
        self.y_pred = y_pred_rbf

        return True

    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive evaluation metrics."""
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Detailed confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        print(f"\n🎯 {model_name} Results:")
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        if cm.shape == (2, 2):
            print(f"\n   Confusion Matrix Breakdown:")
            print(f"   True Negatives (TN): {tn}")
            print(f"   False Positives (FP): {fp}")
            print(f"   False Negatives (FN): {fn}")
            print(f"   True Positives (TP): {tp}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": y_pred,
        }

    def analyze_feature_importance(self):
        """Analyze and display feature importance using linear SVM coefficients."""
        print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("-" * 30)

        # Get feature importance from linear SVM coefficients
        feature_importance = np.abs(self.svm_linear.coef_[0])

        # Create importance ranking
        sorted_idx = np.argsort(feature_importance)[::-1]

        print("Feature Importance Ranking (based on Linear SVM coefficients):")
        print("-" * 55)

        for i, idx in enumerate(sorted_idx):
            importance = feature_importance[idx]
            feature_name = self.features[idx]
            print(f"{i+1:2d}. {feature_name:<25} {importance:.4f}")

        # Store for visualization
        self.feature_importance = feature_importance
        self.feature_ranking = sorted_idx

        return True

    def create_visualizations(self):
        """Create comprehensive visualizations for analysis results."""
        print("\n📊 GENERATING VISUALIZATIONS")
        print("-" * 30)

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # 1. Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = self.results["rbf"]["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Poor (0)", "Good (1)"],
            yticklabels=["Poor (0)", "Good (1)"],
        )
        plt.title("Confusion Matrix (RBF SVM)", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # 2. Prediction Distribution
        plt.subplot(2, 3, 2)
        unique, counts = np.unique(self.y_pred, return_counts=True)
        colors = ["#FF6B6B", "#4ECDC4"]
        bars = plt.bar(["Poor (0)", "Good (1)"], counts, color=colors, alpha=0.8)
        plt.title("Prediction Distribution", fontsize=12, fontweight="bold")
        plt.xlabel("Wine Quality Class")
        plt.ylabel("Number of Predictions")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. Model Comparison
        plt.subplot(2, 3, 3)
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        rbf_values = [self.results["rbf"][m.lower().replace("-", "_")] for m in metrics]
        linear_values = [
            self.results["linear"][m.lower().replace("-", "_")] for m in metrics
        ]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(
            x - width / 2,
            rbf_values,
            width,
            label="RBF SVM",
            color="#FF9FF3",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            linear_values,
            width,
            label="Linear SVM",
            color="#54A0FF",
            alpha=0.8,
        )

        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("Model Performance Comparison", fontsize=12, fontweight="bold")
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.ylim(0, 1.1)

        # 4. Feature Importance (Top 10)
        plt.subplot(2, 3, 4)
        top_10_idx = self.feature_ranking[:10]
        top_10_importance = self.feature_importance[top_10_idx]
        top_10_names = [self.features[i] for i in top_10_idx]

        plt.barh(
            range(len(top_10_names)), top_10_importance, color="#26DE81", alpha=0.8
        )
        plt.yticks(range(len(top_10_names)), top_10_names)
        plt.xlabel("Importance Score")
        plt.title("Top 10 Most Important Features", fontsize=12, fontweight="bold")
        plt.gca().invert_yaxis()

        # 5. Class Distribution Comparison
        plt.subplot(2, 3, 5)
        train_dist = self.train_df[self.target_col].value_counts().sort_index()
        val_dist = self.val_df[self.target_col].value_counts().sort_index()

        x = np.arange(len(train_dist))
        width = 0.35

        plt.bar(
            x - width / 2,
            train_dist.values,
            width,
            label="Training",
            color="#FF7675",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            val_dist.values,
            width,
            label="Validation",
            color="#74B9FF",
            alpha=0.8,
        )

        plt.xlabel("Wine Quality Class")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution Comparison", fontsize=12, fontweight="bold")
        plt.xticks(x, ["Poor (0)", "Good (1)"])
        plt.legend()

        # 6. Performance Metrics Radar Chart (simplified as bar chart)
        plt.subplot(2, 3, 6)
        metrics_values = [
            self.results["rbf"]["accuracy"],
            self.results["rbf"]["precision"],
            self.results["rbf"]["recall"],
            self.results["rbf"]["f1_score"],
        ]

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        bars = plt.bar(metrics, metrics_values, color=colors, alpha=0.8)
        plt.title("RBF SVM Performance Metrics", fontsize=12, fontweight="bold")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

        print("✅ Visualizations generated successfully!")

        return True

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)

        # Dataset Summary
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Training samples: {len(self.train_df):,}")
        print(f"   Validation samples: {len(self.val_df):,}")
        print(f"   Total features: {len(self.features)}")
        print(f"   Target classes: {sorted(self.train_df[self.target_col].unique())}")

        # Model Performance
        print(f"\n🎯 MODEL PERFORMANCE:")
        rbf_results = self.results["rbf"]
        print(f"   Primary Model: SVM with RBF Kernel")
        print(
            f"   Validation Accuracy: {rbf_results['accuracy']:.4f} ({rbf_results['accuracy']*100:.2f}%)"
        )
        print(f"   Precision: {rbf_results['precision']:.4f}")
        print(f"   Recall: {rbf_results['recall']:.4f}")
        print(f"   F1-Score: {rbf_results['f1_score']:.4f}")

        # Feature Analysis
        print(f"\n🔍 KEY INSIGHTS:")
        top_3_idx = self.feature_ranking[:3]
        print(f"   Top 3 Most Important Features:")
        for i, idx in enumerate(top_3_idx, 1):
            feature_name = self.features[idx]
            importance = self.feature_importance[idx]
            print(f"     {i}. {feature_name} (importance: {importance:.4f})")

        # Data Quality Assessment
        missing_total = (
            self.train_df.isnull().sum().sum() + self.val_df.isnull().sum().sum()
        )
        print(f"\n✅ DATA QUALITY:")
        print(f"   Missing values: {missing_total}")
        print(f"   Preprocessing: StandardScaler normalization applied")
        print(
            f"   Data integrity: {'✅ Passed' if missing_total == 0 else '⚠️ Issues detected'}"
        )

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        accuracy = rbf_results["accuracy"]
        if accuracy >= 0.90:
            print("   🌟 Excellent model performance! Ready for production use.")
        elif accuracy >= 0.80:
            print(
                "   ✅ Good model performance. Consider hyperparameter tuning for improvement."
            )
        elif accuracy >= 0.70:
            print(
                "   ⚠️ Moderate performance. Recommend feature engineering or different algorithms."
            )
        else:
            print("   ❌ Low performance. Significant model improvements needed.")

        print(f"\n   Next steps:")
        print(f"   • Consider cross-validation for more robust evaluation")
        print(f"   • Experiment with hyperparameter optimization")
        print(f"   • Try ensemble methods for improved performance")
        print(f"   • Analyze misclassified samples for insights")

        print("\n" + "=" * 60)
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        return True

    def run_complete_analysis(self):
        """Execute the complete wine quality analysis pipeline."""
        try:
            # Execute analysis pipeline
            if not self.load_data():
                return False

            if not self.inspect_data():
                return False

            if not self.preprocess_data():
                return False

            if not self.train_models():
                return False

            if not self.evaluate_models():
                return False

            if not self.analyze_feature_importance():
                return False

            if not self.create_visualizations():
                return False

            if not self.generate_report():
                return False

            print("\n🎉 Analysis completed successfully!")
            return True

        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print("Please check your data files and try again.")
            return False


def main():
    """Main function to run the wine quality analysis."""
    try:
        # Create analyzer instance
        analyzer = WineQualityAnalyzer()

        # Run complete analysis
        success = analyzer.run_complete_analysis()

        if success:
            print("\n✅ All analysis steps completed successfully!")
            print("Check the generated visualizations and report above.")
        else:
            print("\n❌ Analysis failed. Please check the error messages above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
