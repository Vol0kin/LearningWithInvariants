import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

def run_single_experiment(X_train, X_test, y_train, y_test, clf, model_parameters):
    model = GridSearchCV(clf, model_parameters, cv=3, scoring='accuracy', n_jobs=4)
    model.fit(X_train, y_train)
    
    print('Best estimator: ', model.best_estimator_)
    
    y_hat = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_hat)
    
    print('Accuracy: ', accuracy)
    
    return accuracy


def run_multiple_experiments(X_train, y_train, X_test, y_test, clf, model_parameters, seeds):
    accuracies = []
    
    for seed in seeds:
        print(f"Running experiment with random_state={seed}")

        model_parameters['random_state'] = [seed]
        
        acc = run_single_experiment(X_train, X_test, y_train, y_test, clf, model_parameters)
        accuracies.append(acc)
    
    return accuracies


def run_experiment(X, y, seeds, train_sizes, models, models_params):
    accuracies = []
    models_types = []
    used_train_sizes = []
    

    for idx, data_seed in enumerate(seeds):
        print(f"-------------------- EXPERIMENT {idx + 1} --------------------\n\n")
        # Generate test data and training data that will be sampled
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=0.8,
            stratify=y,
            random_state=data_seed
        )
        
        for train_size in train_sizes:
            # Generate training subsamples
            if train_size == 1.0:
                X_train_sub, y_train_sub = X_train, y_train
            else:
                X_train_sub, _, y_train_sub, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=train_size,
                    stratify=y_train,
                    random_state=data_seed
                )

            print(f"\n\n------ Using {train_size} of the training data ------\n\n")
            
            for model, model_params in zip(models, models_params):
                model_type, clf = model
                print(f"Training {model_type} model\n")
                model_accuracies = run_multiple_experiments(
                    X_train_sub,
                    y_train_sub,
                    X_test,
                    y_test,
                    clf,
                    model_params,
                    seeds,
                )

                accuracies.append(model_accuracies)
                models_types.append(model_type)
                used_train_sizes.append(train_size)

                print('\n\n')

    accuracies = np.array(accuracies)
    accuracies_dict = {f'accuracy_{i + 1}': accuracies[:, i] for i in range(accuracies.shape[1])}

    results_df = pd.DataFrame({
        'train_size': used_train_sizes,
        'model': models_types,
        **accuracies_dict,
    })

    return results_df
