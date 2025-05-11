from random import shuffle
import os, json, sqlite3, re, textwrap
from joblib import dump
import autogen
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from database import CervicalCancerDB
from database_Lung import LungCancerDB

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

def supports_multi_output(estimator):
    """Check if the estimator supports multi-output natively."""
    if 'XGB' in type(estimator).__name__:
        return True
    tags = estimator._get_tags()
    return tags.get('multioutput', False)

class RAGLung:
    def __init__(self, db, top_num=2):
        self.db = db
        self.top_num = top_num

        # OAR weights for similarity score calculation
        self.weights = np.array([0.5, 0.4, 0.1, 0, 0])  # Rectum, Bladder, Bowel Bag, Femoral Head L, Femoral Head R

    def calc_similarity_score(self, overlap_ep, overlap_qp):
        similarity_scores = 1 - np.abs(overlap_ep - overlap_qp)
        weighted_scores = similarity_scores * self.weights
        return np.sum(weighted_scores)

    def calc_overlap_percentage(self, patient_name):
        patient_volumes, overlapping_volumes = self.db.get_patient(patient_name)
        overlapping_volumes = np.array(overlapping_volumes[1:])
        patient_volumes = np.array(patient_volumes[2:2+5])
        return overlapping_volumes / patient_volumes

    def retrieve_similar_plans(self, query_patient_name, top_n=2):
        overlap_qp = self.calc_overlap_percentage(query_patient_name)

        scored_patients = []
        for name in self.db.get_all_patient_names():
            if name != query_patient_name:
                overlap_ep = self.calc_overlap_percentage(name)
                score = self.calc_similarity_score(overlap_ep, overlap_qp)
                scored_patients.append((name, overlap_ep, score))

        sorted_patients = sorted(scored_patients, key=lambda x: x[2], reverse=True)
        return sorted_patients[:top_n]
  
    def get_patient_anatomy_str(self, name):
        patient_volumes, overlapping_volumes = self.db.get_patient(name)
        if patient_volumes is None:
            return "Patient not found."
        ov_perc = self.calc_overlap_percentage(name)

        ret = textwrap.dedent(f"""
        - Structures and Volumes:

        | ROIs                       | Volume (cm³)              |
        |----------------------------|---------------------------|
        | PTV                        | {patient_volumes[1]:.2f}  |
        | Bladder                    | {patient_volumes[3]:.2f}  |
        | Rectum                     | {patient_volumes[2]:.2f}  |
        | Bowel Bag                  | {patient_volumes[4]:.2f}  |
        | Femoral Head (L)           | {patient_volumes[5]:.2f}  |
        | Femoral Head (R)           | {patient_volumes[6]:.2f}  |
        | Ring around PTV (R1)       | {patient_volumes[7]:.2f}  |
        | Ring around PTV (R2)       | {patient_volumes[8]:.2f}  |
        | Ring around PTV (R3)       | {patient_volumes[9]:.2f}  |
        | Normal Tissue (NT)         | {patient_volumes[10]:.2f} |

        - Overlapping Volumes with PTV:

        | Structure                  |  Volume (cm³)             | Overlap Percentage |
        |----------------------------|---------------------------| -------------------|
        | Rectum                     | {overlapping_volumes[1]:.2f} | {ov_perc[0]:.2f} |
        | Bladder                    | {overlapping_volumes[2]:.2f} | {ov_perc[1]:.2f} |
        | Bowel Bag                  | {overlapping_volumes[3]:.2f} | {ov_perc[2]:.2f} |
        | Femoral Heads (L & R)      | {overlapping_volumes[4]:.2f}, {overlapping_volumes[5]:.2f} | {ov_perc[3]:.2f}, {ov_perc[4]:.2f} |
        """)
        return ret

    def get_delivery_technique_str(self):
        ret = textwrap.dedent("""
        | Item | Value |
        |-----------------|-----------------|
        | Position        | Prone           |
        | Technique       | VMAT with Dual arcs 181°->179° (CW), 179°->181° (CCW) |
        """)
        return ret

    def get_dose_objective_str(self):
        # all patients shared the same dose objective and constraints
        return textwrap.dedent("""
        | OARs            | Criterion              | Specification|
        |-----------------|--------------------|--------------|
        | Prescribed Dose (PD) | NA                | 5040 cGy delivered in 28 fractions|
        | PTV             | D95 ≥ 100% PD          | At least 95% volume receive 100% of PD|
        | PTV             | Max Dose ≤ 110% PD     | Maximum dose not exceed 110% of PD (5544 cGy); DO NOT use 5544 as Target Max Dose in OptPara |
        | PTV             | Uniformity             | as uniform as possible|
        | Rectum & Bladder| V50 ≤ 50%              | less than 50% volume receive more than 5000 cGy|
        | Bowel Bag       | V45 ≤ 195 cm³          | less than 195 cm³ volume receive more than 4500 cGy|
        | Femoral Head    | V50 ≤ 5%               | less than 5% volume receive more than 5000 cGy|
        """)

    def get_optim_parameters_str(self, patient_name):
        data = self.db.get_optim_parameters(patient_name)  # Fetch the data from the database
        table = "\n"
        table += "| ROI Name | Objective Type | Target cGy | % Volume | Weight |\n"
        table += "|----------|----------------|------------|----------|--------|\n"
        for row in data:
            table += "| {} | {} | {} | {} | {} |\n".format(
                row[1],  # roi_name
                row[2],  # objective_type
                row[3],  # target_cgy
                row[4] if row[4] is not None else 'NA',  # percent_volume
                row[5]  # weight
            )
        return table
    
    def get_trajectory_str(self, patient_name):
        s = self.db.get_trajectory(patient_name)
        if s:
            return s
        else:
            return "NA"

    def print_top_plans(self, query_name, top_plans):
        top_plans = top_plans.copy()

        query_plan = ["Query Patient", np.array((self.calc_overlap_percentage(query_name))), None]
        top_plans.insert(0, query_plan)

        # Extract patient names, OARs overlap percentages, and similarity scores from top_plans
        patient_names = [plan[0] for plan in top_plans]
        oars_overlaps = [plan[1] for plan in top_plans]
        similarity_scores = [plan[2] for plan in top_plans]

        # Create a DataFrame for OARs overlap percentages
        oars_df = pd.DataFrame(oars_overlaps, columns=['Rectum', 'Bladder', 'Bowel_bag', 'Femoral_head_L', 'Femoral_head_R'])

        # Add patient names and similarity scores to the DataFrame
        oars_df.insert(0, 'Patient-Name', patient_names)
        oars_df['Similarity Score'] = similarity_scores

        # Print the DataFrame
        print("\033[93mReference Plans based on OARs' Overlap Percentages Similarity:\033[0m")
        print(oars_df)

    def get_all_patient_info(self):
        patient_names = self.db.get_all_patient_names()
        info = f"Cervical Cancer Treatment Plans Database\n"
        info += f"All Patients have a same dose objectives and delivery technique as follows:\n{self.get_dose_objective_str()}\n{self.get_delivery_technique_str()}\nPatients In Database:\n"

        for name in patient_names:
            if 'A' not in name and "ZXY" not in name:
                info += f"\n\nPatient Name: {name}\n{self.get_patient_anatomy_str(name)}\n{self.get_optim_parameters_str(name)}"

        return info

    @staticmethod
    def unit_test():
        db = CervicalCancerDB()
        db.get_all_patient_names()

        rag = RAGLung(db, 3)

        all_pat_info = rag.get_all_patient_info()
        print(all_pat_info)

        query_name = "A"
        print(rag.get_patient_anatomy_str(query_name))
        # print(self.get_optim_parameters_str(query_name))

        top_plans = rag.retrieve_similar_plans(query_name, top_n=10)
        rag.print_top_plans(query_name, top_plans)

class RAGLung:
    def __init__(self, db: LungCancerDB, top_num=2):
        self.db = db
        self.top_num = top_num

    def get_patient_anatomy_str(self, name):
        volume, dist = self.db.get_patient(name)
        volume = f'Struct Volumes (cc): PTV {volume[1]}, ESOPHAGUS {volume[2]}, HEART {volume[3]}, LUNG_L {volume[4]}, LUNG_R {volume[5]}, CORD {volume[6]}.'
        dist = f'Distances to PTV: ESOPHAGUS {dist[1]}, HEART {dist[2]}, LUNG_L {dist[3]}, LUNG_R {dist[4]}, CORD {dist[5]}.'
        return volume + '\n' + dist 

    def get_ref_plans(self, patient_name):
        names = self.db.get_all_patient_names()
        names = [name for name in names if patient_name not in name]
        shuffle(names)
        names = names[:self.top_num]

        plans = [self.get_OptPara_DoseOut(name) for name in names]

        return '\n'.join(plans)

    def get_OptPara_DoseOut(self, name):
        anatomy = self.get_patient_anatomy_str(name)
        name, optPara, dose_out = self.db.get_best_iteration(name)[0]
        return textwrap.dedent(f"""\
### Ref Plan: {name}
Anatomy:
{anatomy}

{optPara}

{dose_out}
""")

class CervicalCancerSimilarity:
    def __init__(self, db: CervicalCancerDB, top_num=3):
        self.db = db
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.top_num = top_num

    def get_structural_features(self, name=None):
        if name is not None:
            patient_names = [name]
        else:
            patient_names = self.db.get_all_patient_names()
            # drop "A" and "ZXY" patients
            patient_names = [name for name in patient_names if 'A' not in name and 'ZXY' not in name]

        rows = []
        for name in patient_names:
            patient_volumes, overlapping_volumes = self.db.get_patient(name)
            row = {
                'Name': name,
                'PTV_Volume': patient_volumes[1], 'Rectum_Volume': patient_volumes[2], 'Bladder_Volume': patient_volumes[3],
                'Bowel_Bag_Volume': patient_volumes[4], 'Femoral_Head_L_Volume': patient_volumes[5], 'Femoral_Head_R_Volume': patient_volumes[6],
                'R1_Volume': patient_volumes[7], 'R2_Volume': patient_volumes[8], 'R3_Volume': patient_volumes[9],
                'NT_Volume': patient_volumes[10], 'Rectum_Overlap': overlapping_volumes[1], 'Bladder_Overlap': overlapping_volumes[2],
                'Bowel_Bag_Overlap': overlapping_volumes[3], 'Femoral_Head_Overlap_L': overlapping_volumes[4],
                'Femoral_Head_Overlap_R': overlapping_volumes[5]
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def get_optim_param(self, name=None):
        query = '''
        SELECT patient_name, roi_name, objective_type, target_cgy, percent_volume, weight
        FROM plan_optimization_parameters
        ORDER BY patient_name, roi_name, objective_type
        '''
        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()

        summary_data = [
            {
                'Name': patient_name,
                'Column': f"{roi_name.lower()}_{objective_type.lower().replace(' ', '_')}",
                'Value': int(target_cgy)
            }
            for patient_name, roi_name, objective_type, target_cgy, percent_volume, weight in result
        ]

        df = pd.DataFrame(summary_data)
        df = df.groupby(['Name', 'Column']).Value.agg(list).reset_index()
        df = df.pivot(index='Name', columns='Column', values='Value').reset_index()
        df = df.sort_values('Name').reset_index(drop=True)
        df_non_dvh, df_dvh = df.loc[:, ~df.columns.str.contains('dvh')].applymap(lambda x: x[0] if isinstance(x, list) else x), df.loc[:, df.columns.str.contains('dvh')]

        common_dvh_target_cgy = {
            col: list(set.intersection(*df_dvh[col].apply(set)))
            for col in df_dvh.columns
        }

        new_dvh_df = pd.DataFrame([
            {
                'Name': patient_name,
                'Column': f"{roi_name.lower()}_{objective_type.lower().replace(' ', '_')}_{com_target_cgy}",
                'Value': percent_volume
            }
            for patient_name, roi_name, objective_type, target_cgy, percent_volume, weight in result
            for dvh_col in common_dvh_target_cgy.keys()
            for com_target_cgy in common_dvh_target_cgy[dvh_col]
            if roi_name.lower() in dvh_col and objective_type.lower().replace(" ", '_') in dvh_col and com_target_cgy == target_cgy
        ])
        new_dvh_df = new_dvh_df.pivot(index='Name', columns='Column', values='Value')

        df = pd.merge(df_non_dvh, new_dvh_df, on='Name')
        if name is not None:
            df = df[df['Name'] == name]
        
        return df

    def cal_similarity_matrix(self, df):
        # Create a new DataFrame that excludes the 'Name' column
        df_without_name = df.drop('Name', axis=1)
        
        # Normalize the data
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_without_name), columns=df_without_name.columns)
        
        similarity_matrix = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                # Use df_normalized for the subtraction operation
                distance = np.linalg.norm(df_normalized.iloc[i].values - df_normalized.iloc[j].values)
                similarity = 1 / (1+distance)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        # create a df with column and row labels as patient names
        # print("Similarity Matrix:")
        similarity_df = pd.DataFrame(similarity_matrix.round(5), columns=df['Name'].values, index=df['Name'].values)
        # print(similarity_df)
        return similarity_df

    def prepare_data(self):
        # X
        df_X = self.get_structural_features()
        self.X_names = df_X.columns[1:]          

        # y
        df_y = self.get_optim_param()
        self.y_names = df_y.columns[1:]
        # TODO
        # self.y_names = self.y_names[:4]

        # merge X and y
        df = pd.merge(df_X, df_y, on='Name')
        return df

    def train_models(self, data_df):
        data_df = data_df.drop(columns=['Name'])
        X = data_df[self.X_names]
        y = data_df[self.y_names]
        # Scale X and y separately
        X = pd.DataFrame(self.scaler_X.fit_transform(X), columns=X.columns)
        y = pd.DataFrame(self.scaler_y.fit_transform(y), columns=y.columns)
        # self.poly = PolynomialFeatures(degree=4, include_bias=False)
        # X = self.poly.fit_transform(X)

        # find the best model and best hyper parameters
        model_list = {
            # 'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42),
            # 'KNN': KNeighborsRegressor(),
        }

        param_grid = {
            # 'RandomForest': {'n_estimators': [50, 100, 200, 500], 'max_depth': [None, 10, 20, 30, 50], 'min_samples_split': [2, 5, 7]},
            'XGBoost': {'n_estimators': [50, 100, 200, 500], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth': [3, 6, 9, 12, 20, 50]},
            # 'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
        }

        param_grid = {
            'XGBoost': {
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                'n_estimators': [50, 100, 200, 300, 400, 500, 600],
                'max_depth': [3, 6, 9, 12, 20, 30, 40, 50, 60],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.5, 0.7, 1.0],
                'colsample_bytree': [0.5, 0.7, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4]
            }
        }

        best_score = float('-inf')
        best_model = None
        best_params = None
        cv_results = {}
        for model_name, model in model_list.items():
            print(f"Training {model_name}...")
            if not supports_multi_output(model):
                model = MultiOutputRegressor(model)
            grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X, y)
            cv_results[model_name] = grid_search.cv_results_['mean_test_score']
            mean_score = grid_search.best_score_
            if mean_score > best_score:
                best_score = mean_score
                best_model = model_name
                best_params = grid_search.best_params_
        
        for model_name, scores in cv_results.items():
            print(f"{model_name} CV scores: {scores}")

        # fit all data with the best model and best hyper parameters
        best_model_instance = model_list[best_model].set_params(**best_params)
        best_model_instance.fit(X, y)

        # predict y
        y_pred = best_model_instance.predict(X)
        y_true = y.values

        # Print predicted and ground truth y
        print("Predicted y:", y_pred)
        print("Ground truth y:", y_true)

        return best_model_instance

    def predict_opt_params(self, query_struct_df, model):
        query_struct_df.drop(columns=['Name'], inplace=True)
        query_struct_df = pd.DataFrame(self.scaler_X.transform(query_struct_df), columns=query_struct_df.columns)

        predicted_opt_params = model.predict(query_struct_df)
        predicted_opt_params = self.scaler_y.inverse_transform(predicted_opt_params)

        predicted_opt_params_dict = dict(zip(self.y_names, predicted_opt_params[0]))
        return predicted_opt_params_dict

    def measure_similarity(self, predicted_opt_params, data_df):
        similarities = []
        for i, row in data_df.iterrows():
            db_opt_params = row[self.y_names].values
            pred_opt_params = np.array([predicted_opt_params[y_name] for y_name in self.y_names])

            # Normalize the parameters before calculating the distance
            db_opt_params = self.scaler_y.transform([db_opt_params])[0]
            pred_opt_params = self.scaler_y.transform([pred_opt_params])[0]

            similarity = np.linalg.norm(db_opt_params - pred_opt_params)  # Euclidean distance
            similarities.append((row['Name'], similarity))
        return sorted(similarities, key=lambda x: x[1]) 

    def find_most_similar_patients(self, query_struct_df, top_n=2):
        data_df = self.prepare_data()
        model = self.train_models(data_df)
        predicted_opt_params = self.predict_opt_params(query_struct_df, model)

        similarities = self.measure_similarity(predicted_opt_params, data_df)
        return similarities[:top_n]

    def precision_recall_metrics(self, predicted):
        gt_similarity_df = self.cal_similarity_matrix(self.get_structural_features())

        # Initialize counters and lists for metrics
        tp = 0
        fp = 0
        fn = 0
        average_precisions = []

        # Iterate over all patients in the predicted dictionary
        for patient, predicted_similar in predicted.items():
            predicted_similar = set(predicted_similar)  # Top N predictions are already considered
            actual_similar = set(gt_similarity_df.loc[patient].nlargest(len(predicted_similar)).index)

            # Calculate intersection and differences for TP, FP, FN
            tp += len(predicted_similar & actual_similar)
            fp += len(predicted_similar - actual_similar)
            fn += len(actual_similar - predicted_similar)

            # Calculate precision at each relevant retrieved item for MAP@N
            precisions = []
            relevant_hits = 0
            sorted_predicted = list(predicted_similar)
            for i, p in enumerate(sorted_predicted):
                if p in actual_similar:
                    relevant_hits += 1
                    current_precision = relevant_hits / (i + 1)
                    precisions.append(current_precision)
                    # print(f"Item: {p}, Rank: {i+1}, Precision at this rank: {current_precision}")

            average_precision = sum(precisions) / relevant_hits if relevant_hits > 0 else 0
            # print(f"Average Precision for patient {patient}: {average_precision}")
            average_precisions.append(average_precision)

        # Calculate precision and recall
        precision_N = tp / (tp + fp) if tp + fp > 0 else 0
        recall_N = tp / (tp + fn) if tp + fn > 0 else 0

        # Calculate F1-Score@N
        if precision_N + recall_N > 0:
            f1_score_N = 2 * (precision_N * recall_N) / (precision_N + recall_N)
        else:
            f1_score_N = 0

        # Calculate MAP@N
        map_N = sum(average_precisions) / len(average_precisions) if average_precisions else 0
        print("Precision@N:", precision_N)
        print("Recall@N:", recall_N)
        print("F1-Score@N:", f1_score_N)
        print("MAP@N:", map_N)

        return precision_N, recall_N, f1_score_N, map_N

    def overlap_similarity(self, df, rag):
        most_similar_patients = {}
        for query_name in df['Name']: 
            top_plans = rag.retrieve_similar_plans(query_name, top_n=self.top_num)
            most_similar_patients[query_name] = [p[0] for p in top_plans] 
        self.precision_recall_metrics(most_similar_patients)

    def _calculate_similarity(self, df, n_components, method):
        df = df[self.X_names].join(df['Name'])
        df_noName = df.drop('Name', axis=1)

        # Standardize the data
        scaler = StandardScaler()
        df_noName = pd.DataFrame(scaler.fit_transform(df_noName), columns=df_noName.columns)

        # Perform dimensionality reduction
        if method == 'PCA':
            from sklearn.decomposition import PCA
            model = PCA(n_components=n_components)
        elif method == 'tSNE':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=n_components, perplexity=df_noName.shape[0]//3)
        elif method == 'UMAP':
            import umap
            model = umap.UMAP(n_components=n_components, n_neighbors=df_noName.shape[0]//3)

        df_reduced = model.fit_transform(df_noName)
        df_reduced = pd.DataFrame(df_reduced, columns=[f'{method}{i+1}' for i in range(n_components)]) 
        df_reduced = df_reduced.join(df['Name'])

        # Calculate the similarity matrix in the reduced space
        sim = self.cal_similarity_matrix(df_reduced)

        # Find the top n most similar patients for each patient
        most_similar_patients = {}
        for patient in sim.index:
            most_similar_patients[patient] = sim.loc[patient].nlargest(self.top_num).index.tolist()

        # Calculate precision and recall
        self.precision_recall_metrics(most_similar_patients)

    def pca_similarity(self, df, n_components):
        self._calculate_similarity(df, n_components, 'PCA')

    def tSNE_similarity(self, df, n_components):
        self._calculate_similarity(df, n_components, 'tSNE')

    def UMAP_similarity(self, df, n_components):
        self._calculate_similarity(df, n_components, 'UMAP')

    @staticmethod 
    def unit_test():
        top_num = 1 

        db = CervicalCancerDB()
        sm = CervicalCancerSimilarity(db, top_num=top_num)
        df = sm.prepare_data()

        print('pca_similarity')
        sm.pca_similarity(df, n_components=3)
        print('tSNE_similarity')
        sm.tSNE_similarity(df, n_components=3)
        print('UMAP_similarity')
        sm.UMAP_similarity(df, n_components=3)
        print('overlap_similarity')
        rag = RAGLung(db, top_num)
        sm.overlap_similarity(df, rag)

        query_name = 'B'
        query_struct_df = sm.get_structural_features(query_name)

        most_similar_patients = sm.find_most_similar_patients(query_struct_df)
        print(most_similar_patients)

if __name__ == '__main__':
    # RAG.unit_test()
    CervicalCancerSimilarity.unit_test()