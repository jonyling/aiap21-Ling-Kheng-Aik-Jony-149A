import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

def build_pipelines(config):
    """
    Build imblearn Pipelines with preprocessing, resampling, and classifiers.
    """
    numerical_features = config.get("numerical_features", ['Temperature_x_Humidity', 'CO2_sum', 'MetalOxideSensor_sum'])
    nominal_features = config.get("nominal_features", ['HVAC Operation Mode'])
    ordinal_features = config.get("ordinal_features", ['Time of Day', 'Session_ID_Bands', 'CO_GasSensor'])

    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    nominal_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[
            ['morning', 'afternoon', 'evening', 'night'],
            ['1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000', '8000-9000', '9000-10000'],
            ['extremely low', 'low', 'medium', 'high', 'extremely high']
        ], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features),
        ],
        remainder='drop',
        n_jobs=-1,
        verbose_feature_names_out=False
    )

    smote = SMOTE(sampling_strategy={2: 2000}, random_state=42)
    undersample = RandomUnderSampler(sampling_strategy={0: 2000}, random_state=42)

    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, n_jobs=-1)
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('undersampler', undersample),
            ('classifier', model)
        ])

    return pipelines

def evaluate_models(pipelines, X_train, X_test, y_train, y_test, config):
    """
    Fits and evaluates a dictionary of pipelines.
    """
    results = {}
    
    for name, pipeline in pipelines.items():
        print(f"Fitting {name}...")
        
        # Fit the entire pipeline (preprocessing, resampling, and model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        results[name]['cv_f1_macro'] = cv_scores.mean()

        # Save the trained pipeline
        os.makedirs('models', exist_ok=True)
        joblib.dump(pipeline, f'models/{name}.pkl')
        
        print(f"\nClassification Report ({name}):")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"CV Macro F1: {metrics['cv_f1_macro']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

    param_grids = {
        'DecisionTree': {
            'classifier__max_depth': [3, 5, 7, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 10, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'LogisticRegression': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'saga']
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__subsample': [0.6, 0.8, 1.0]
        }
    }

    for name, pipeline in pipelines.items():
        print(f"Fitting {name} with hyperparameter tuning...")
        param_grid = param_grids.get(name, {})
        if param_grid:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=10,  # Number of random combinations to try
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1_macro',
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            print(f"Best parameters for {name}: {search.best_params_}")
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)

        y_pred = best_pipeline.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'cv_f1_macro': search.best_score_ if param_grid else cross_val_score(
                best_pipeline, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_macro', n_jobs=-1
            ).mean()
        }

        os.makedirs('models', exist_ok=True)
        joblib.dump(best_pipeline, f'models/{name}.pkl')
        print(f"\nClassification Report ({name}):")
        print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High']))

    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"CV Macro F1: {metrics['cv_f1_macro']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

    return pipelines, results

# =============================================================================
# 4. Feature Importance Analysis
# =============================================================================

def get_preprocessor_feature_names(fitted_preprocessor):
    """
    Extracts feature names after preprocessing from a fitted ColumnTransformer.
    """
    return fitted_preprocessor.get_feature_names_out()

def analyze_feature_importance(pipeline, X_train, y_train, model_name, class_labels):
    """
    Analyzes and displays feature importances for tree-based models.
    """
    print(f"\n=================================================")
    print(f"Feature Importance Analysis for {model_name}")
    print(f"=================================================\n")

    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = get_preprocessor_feature_names(preprocessor)    
    classifier = pipeline.named_steps['classifier']
    
    # Check if the classifier has a feature_importances_ attribute
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"Top 10 most important features for {model_name}:\n")
        print(feature_importance_df.head(10))
        
        # Plotting the feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
        plt.title(f'Top 10 Feature Importances for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

    else:
        print(f"Warning: {model_name} does not have a feature_importances_ attribute.")

def analyze_logistic_regression_coefficients(pipeline, X_train, y_train, class_labels):
    """
    Analyzes and displays coefficients for each class in a Logistic Regression model.
    This helps to understand which features contribute positively or negatively to each class.
    """
    print(f"\n=================================================")
    print(f"Coefficient Analysis for Logistic Regression")
    print(f"=================================================\n")

    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = get_preprocessor_feature_names(preprocessor)
    classifier = pipeline.named_steps['classifier']
    
    # Check if the classifier has a coef_ attribute
    if hasattr(classifier, 'coef_'):
        coefficients = classifier.coef_
        num_classes = coefficients.shape[0]
        
        for i in range(num_classes):
            print(f"Top 10 features for '{class_labels[i]}' (Class {i}):\n")
            
            class_coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients[i]})
            
            # Sort by absolute value to find most influential features
            class_coef_df['abs_coefficient'] = class_coef_df['coefficient'].abs()
            class_coef_df = class_coef_df.sort_values(by='abs_coefficient', ascending=False)

            print(class_coef_df.drop('abs_coefficient', axis=1).head(10))
            
            # Plotting the coefficients
            plt.figure(figsize=(12, 8))
            sns.barplot(x='coefficient', y='feature', data=class_coef_df.head(10))
            plt.title(f'Top 10 Feature Coefficients for class "{class_labels[i]}"')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.show()
            
            print("-------------------------------------------------------------------\n")
    else:
        print("Warning: Logistic Regression model does not have a coef_ attribute.")
        
def main():
    """
    Main function to train and evaluate models.
    """
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load preprocessed data
    df = pd.read_csv('data/processed/preprocessed_data.csv')
    
    # Prepare features and target
    X = df.drop(['Activity Level'], axis=1)
    y = df['Activity Level'].replace({'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build and evaluate pipelines
    pipelines = build_pipelines(config)
    pipelines, results = evaluate_models(pipelines, X_train, X_test, y_train, y_test, config)

    # Analyze feature importance
    class_labels = ['Low Activity', 'Moderate Activity', 'High Activity']
    analyze_feature_importance(pipelines['RandomForest'], X_train, 'RandomForest', class_labels)
    analyze_feature_importance(pipelines['XGBoost'], X_train, 'XGBoost', class_labels)
    analyze_logistic_regression_coefficients(pipelines['LogisticRegression'], X_train, class_labels)

    print("Model training and evaluation completed.")

if __name__ == "__main__":
    main()