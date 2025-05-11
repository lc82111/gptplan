from copy import copy, deepcopy
from scipy import stats
from scipy.ndimage import gaussian_filter
import csv
import glob
import seaborn as sns
import pandas as pd
from pathlib import Path
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from termcolor import colored
import io, re, os, json, sys, pickle
from typing import List, Literal, Union
import pandas as pd
from skimage import measure

import portpy.photon as pp
from portpy.photon.plan import Plan
from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon.clinical_criteria import ClinicalCriteria

import cvxpy as cp
from tqdm import tqdm

class JSON2Markdown_back:
    """convert json to markdown and vice versa"""

    def __init__(self, optim_param_json_fn):
        # Parse JSON strings files into Python dictionaries
        self.op_dict = json.load(open(optim_param_json_fn))

        self.structure_defs = {
            "LUNGS_NOT_GTV": "(LUNG_L | LUNG_R) - GTV",
            "RIND_0": "(PTV+5) - PTV",
            "RIND_1": "(PTV+10) - (PTV + 5)",
            "RIND_2": "(PTV+30) - (PTV + 10)",
            "RIND_3": "(PTV+50) - (PTV + 30)",
            "RIND_4": "(PTV + inf) - (PTV + 50)"
        }

    def json_to_markdown(self, optim_param_json_fn=None):
        if optim_param_json_fn:
            op_dict = json.load(open(optim_param_json_fn))  
        else:
            op_dict = self.op_dict

        # Create lists to store data
        data = []

        # Process objective functions from optim_param json
        for obj in op_dict['objective_functions']:
            data.append({
                'ROI Name': obj.get('structure_name', 'NA'),
                'Objective Type': obj['type'],
                'Target Gy': obj.get('dose_gy', 'NA'),
                '% Volume': 'NA',
                'Weight': obj.get('weight', 'NA')
            })

        # Process constraints from optim_param json
        for constraint in op_dict.get('constraints', []):
            if constraint['type'] in ['max_dose', 'mean_dose']:
                data.append({
                    'ROI Name': constraint['parameters']['structure_name'],
                    'Objective Type': constraint['type'],
                    'Target Gy': constraint['constraints'].get('limit_dose_gy', 'NA'),
                    '% Volume': 'NA',
                    'Weight': 'NA'
                })
            elif constraint['type'] == 'dose_volume_V':
                data.append({
                    'ROI Name': constraint['parameters']['structure_name'],
                    'Objective Type': constraint['type'],
                    'Target Gy': constraint['parameters']['dose_gy'],
                    '% Volume': constraint['constraints'].get('limit_volume_perc', 'NA'),
                    'Weight': 'NA'
                })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Sort DataFrame by ROI Name
        df = df.sort_values('ROI Name')

        # Step 1: Define a custom sort function
        def sort_roi(roi_name):
            if roi_name.startswith('PTV'):
                return (1, roi_name)  # PTVs first
            if roi_name.startswith('GTV'):
                return (2, roi_name)  # PTVs first
            elif roi_name.startswith('RIND'):
                return (4, roi_name)
            elif roi_name.startswith('SKIN'):
                return (5, roi_name)
            elif roi_name.startswith('NA'):
                return (6, roi_name)
            else:
                return (3, roi_name)  # OARs in the middle

        def sort_type(objective_type):
            order = ['quadratic-overdose', 'linear-overdose', 'quadratic-underdose',
                     'quadratic', 'max_dose', 'mean_dose', 'dose_volume_V']
            try:
                return order.index(objective_type)
            except ValueError:
                return len(order)  # Place unknown values at the end

        # Step 2: Apply the custom sort function to create a new sort key column
        df['SortROI'] = df['ROI Name'].apply(sort_roi)
        df['SortType'] = df['Objective Type'].apply(sort_type)

        # Step 3: Sort the DataFrame using the new sort key column
        df_sorted = df.sort_values(['SortROI', 'SortType'])

        # Step 4: Drop the custom sort key column if no longer needed
        df_sorted = df_sorted.drop(columns=['SortROI', 'SortType'])

        # Convert DataFrame to Markdown
        markdown = df_sorted.to_markdown(index=False)

        return markdown

    def markdown_to_json(self, markdown_table):
        if 'OptPara' in markdown_table:
            # remove the first line 
            markdown_table = '\n'.join(markdown_table.split('\n')[1:])

        # Use pandas to read the Markdown table
        df = pd.read_table(io.StringIO(markdown_table), sep='|', skipinitialspace=True)
        # Clean up the column names and drop any empty columns
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how='all')
        # Remove the separator row (usually the second row in a markdown table)
        df = df.drop(df.index[df.iloc[:, 0].str.contains('-{3,}', na=False)]).reset_index(drop=True)
        # Remove any remaining rows that are all NaN or contain only separators
        df = df[~df.apply(lambda row: row.astype(str).str.contains('-{3,}').all() or row.isna().all(), axis=1)]
        # Reset the index after dropping rows
        df = df.reset_index(drop=True)

        if 'Target cGy' in df.columns:
            # change the column 'Target cGy' to 'Target Gy'
            df = df.rename(columns={'Target cGy': 'Target Gy'})
            # Convert 'Target Gy' column values from string to float, divide by 100, and convert back to string
            df['Target Gy'] = df['Target Gy'].apply(lambda x: str(float(x)/100) if x.strip().replace('.', '', 1).isdigit() else x)

        for _, row in df.iterrows():
            roi_name = row['ROI Name'].strip()
            type = row['Objective Type'].strip()
            target = row['Target Gy'].strip()
            volume = row['% Volume'].strip()
            weight = row['Weight'].strip()

            # convert target, volume, weight to float
            if target.replace('.', '').isdigit():
                target = float(target)
            if volume.replace('.', '').isdigit():
                volume = float(volume)
            if weight.replace('.', '').isdigit():
                weight = float(weight)

            if type in ['quadratic-overdose', 'linear-overdose', 'quadratic-underdose', 'quadratic', 'smoothness-quadratic']:
                # This is an objective function in optim_param json
                obj = {
                    "type": type,
                    "weight": weight
                }
                if roi_name != 'NA':
                    obj["structure_name"] = roi_name
                    if roi_name in self.structure_defs:
                        obj["structure_def"] = self.structure_defs[roi_name]
                if target != 'NA':
                    obj["dose_gy"] = target

                # Update or add to objective_functions
                if 'smoothness' not in type:
                    existing_obj = next((item for item in self.op_dict['objective_functions'] 
                                        if item['type'] == type and item.get('structure_name') == roi_name), None)
                else:
                    existing_obj = next((item for item in self.op_dict['objective_functions'] 
                                        if item['type'] == type), None)
                
                if existing_obj:
                    existing_obj.update(obj)
                else:
                    self.op_dict['objective_functions'].append(obj)

            elif type in ['max_dose', 'mean_dose', 'dose_volume_V']:
                # This is a constraint in optim_param json
                cst = {
                    "type": type,
                    "parameters": {
                        "structure_name": roi_name
                    },
                    "constraints": {}
                }
                if roi_name in self.structure_defs:
                    cst["parameters"]["structure_def"] = self.structure_defs[roi_name]
                if type in ['max_dose', 'mean_dose']:
                    cst["constraints"]["limit_dose_gy"] = target
                elif type == 'dose_volume_V':
                    cst["parameters"]["dose_gy"] = target
                    cst["constraints"]["limit_volume_perc"] = (volume)

                # Update or add 
                existing_cst = next((item for item in self.op_dict['constraints'] 
                                     if item['type'] == type and item['parameters']['structure_name'] == roi_name), None)
                if existing_cst:
                    existing_cst.update(cst)
                else:
                    self.op_dict['constraints'].append(cst)
            
            else:
                raise ValueError(f"Unknown Objective Type: {type}")

        return json.dumps(self.op_dict)

def calculate_score(eval_df):
    score = 0
    for index, row in eval_df.iterrows():
        constraint = row['constraint']
        goal = row['Goal']
        plan_value = row['Plan Value']
        
        if constraint == 'max_dose' or constraint == 'mean_dose' or 'V(' in constraint:
            score += goal - plan_value
        elif constraint == 'D(95.0%)':
            if plan_value < 56:
                return -np.inf, -np.inf 
            score += plan_value - goal
        elif constraint == 'CI':
            ci = plan_value
            score += plan_value - goal
        elif constraint == 'HI':
            score += goal - plan_value
        else:
            raise ValueError(f"Unknown Constraint: {constraint}")
    
    return score, ci 

def extract_numeric(value):
    # Function to extract numeric values from mixed type columns
    if isinstance(value, (int, float)):
        return value
    match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
    if match:
        return float(match[0])
    return np.nan

class JSON2Markdown:
    """convert json to markdown and vice versa"""

    def __init__(self):
        # Parse JSON strings files into Python dictionaries
        # self.op_dict = json.load(open(optim_param_json_fn))

        self.structure_defs = {
            "LUNGS_NOT_GTV": "(LUNG_L | LUNG_R) - GTV",
            "RIND_0": "(PTV+5) - PTV",
            "RIND_1": "(PTV+10) - (PTV + 5)",
            "RIND_2": "(PTV+30) - (PTV + 10)",
            "RIND_3": "(PTV+50) - (PTV + 30)",
            "RIND_4": "(PTV + inf) - (PTV + 50)"
        }

    def markdown_to_json(self, markdown_table):
        if 'OptPara' in markdown_table:
            # remove the first line 
            markdown_table = '\n'.join(markdown_table.split('\n')[1:])

        # Use pandas to read the Markdown table
        df = pd.read_table(io.StringIO(markdown_table), sep='|', skipinitialspace=True)
        # Clean up the column names and drop any empty columns
        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how='all')
        # Remove the separator row (usually the second row in a markdown table)
        df = df.drop(df.index[df.iloc[:, 0].str.contains('-{3,}', na=False)]).reset_index(drop=True)
        # Remove any remaining rows that are all NaN or contain only separators
        df = df[~df.apply(lambda row: row.astype(str).str.contains('-{3,}').all() or row.isna().all(), axis=1)]
        # Reset the index after dropping rows
        df = df.reset_index(drop=True)

        # new opt dict 
        op_dict = {
            "prescription_gy": 60,
            "objective_functions": [],
            "constraints": []
        }

        for _, row in df.iterrows():
            roi_name = row['ROI Name'].strip()
            type = row['Objective Type'].strip()
            target = row['Target Gy'].strip()
            volume = row['% Volume'].strip()
            weight = row['Weight'].strip()

            # convert target, volume, weight to float
            if target.replace('.', '').isdigit():
                target = float(target)
            if volume.replace('.', '').isdigit():
                volume = float(volume)
            if weight.replace('.', '').isdigit():
                weight = float(weight)

            if type in ['quadratic-overdose', 'linear-overdose', 'quadratic-underdose', 'quadratic', 'smoothness-quadratic']:
                # This is an objective function in optim_param json
                obj = {
                    "type": type,
                    "weight": weight
                }
                if roi_name != 'NA':  # type smoothness-quadratic has roi_name as NA
                    obj["structure_name"] = roi_name
                    if roi_name in self.structure_defs:
                        obj["structure_def"] = self.structure_defs[roi_name]

                if target != 'NA':
                    obj["dose_gy"] = target

                # objective_functions
                op_dict['objective_functions'].append(obj)

            elif type in ['max_dose', 'mean_dose', 'dose_volume_V']:
                # This is a constraint in optim_param json
                cst = {
                    "type": type,
                    "parameters": {
                        "structure_name": roi_name
                    },
                    "constraints": {}
                }
                if roi_name in self.structure_defs:
                    cst["parameters"]["structure_def"] = self.structure_defs[roi_name]

                if type in ['max_dose', 'mean_dose']:
                    cst["constraints"]["limit_dose_gy"] = target
                elif type == 'dose_volume_V':
                    cst["parameters"]["dose_gy"] = target
                    cst["constraints"]["limit_volume_perc"] = (volume)

                # Update or add 
                op_dict['constraints'].append(cst)
            
            else:
                raise ValueError(f"Unknown Objective Type: {type}")

        return json.dumps(op_dict)

class MyOptimization(pp.Optimization):
    def __init__(self, my_plan: Plan, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None):
        super().__init__(my_plan, inf_matrix, clinical_criteria, opt_params, vars)
 
    def add_single_dvh_constraint(self, struct, dose_gy, limit_vol_perc, M=50):
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']
        n_frac = self.my_plan.get_num_of_fractions()
        cc = st.get_opt_voxels_volume_cc(struct)  # the volume of each voxel in cubic centimeters (cc)
        idx = st.get_opt_voxels_idx(struct)
        vol_frac = self.my_plan.structures.get_fraction_of_vol_in_calc_box(struct) # fraction of volume in calculation box
        
        # Create a binary variable for the single DVH constraint
        b = cp.Variable(len(st.get_opt_voxels_idx(struct)), boolean=True)
        
        # Define the constraint based on provided parameters
        constraint = [
            A[idx, :] @ x <= dose_gy / n_frac + b * M / n_frac,
            b @ cc <= (limit_vol_perc / vol_frac) / 100 * sum(cc)
        ]
        
        # Add the constraint to the optimization problem
        self.add_constraints(constraints=constraint)

    def create_cvxpy_problem(self):
        """ It runs optimization to create optimal plan based upon clinical criteria
        :return: cvxpy problem object """
        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        x = self.vars['x']
        obj = self.obj
        constraints = self.constraints

        # self.prescription_gy = opt_params['prescription_gy']

        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []

        A = inf_matrix.A
        num_fractions = clinical_criteria.get_num_of_fractions()
        st = inf_matrix

        # Construct optimization problem

        # 1. first we check is there any dose_volume_V constraint
        dvh_constraint_exists = True if any([c['type'] == 'dose_volume_V' for c in opt_params_constraints]) else False 
        print('dvh_constraint_exists', dvh_constraint_exists)

        # Generating objective functions
        print('Objective Start')
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose' or obj_funcs[i]['type'] == 'linear-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        print(f'Passing {struct} as no opt voxels')
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    # change quadratic to linear to reduce computation if dvhs are present
                    if dvh_constraint_exists:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum(dO))]
                        print(f'Adding Linear Overdose: {obj_funcs[i]}')
                    else:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                        print(f'Adding Quadratic Overdose: {obj_funcs[i]}')
                    constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= dose_gy + dO]
                else:
                    print(f'Passing {obj_funcs[i]} as no struct')
            elif obj_funcs[i]['type'] == 'quadratic-underdose' or obj_funcs[i]['type'] == 'linear-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        print(f'Passing {struct} as no opt voxels')
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    if dvh_constraint_exists:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * 0.1 * cp.sum(dU))]  # change quadratic to linear
                        print(f'Adding Linear Underdose: {obj_funcs[i]}')
                    else:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                        print(f'Adding Quadratic Underdose: {obj_funcs[i]}')
                    constraints += [A[st.get_opt_voxels_idx(struct), :] @ x >= dose_gy - dU]
                else:
                    print(f'Passing {obj_funcs[i]} as no struct')
            elif obj_funcs[i]['type'] == 'quadratic' or obj_funcs[i]['type'] == 'linear':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        print(f'Passing {struct} as no opt voxels')
                        continue
                    if dvh_constraint_exists:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum(A[st.get_opt_voxels_idx(struct), :] @ x))]
                        print(f'Adding Linear: {obj_funcs[i]}')
                    else:
                        obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(A[st.get_opt_voxels_idx(struct), :] @ x))] # change quadratic to linear
                        print(f'Adding Quadratic: {obj_funcs[i]}')
                else:
                    print(f'Passing {obj_funcs[i]} as no struct')
            elif obj_funcs[i]['type'] == 'smoothness-quadratic':
                [Qx, Qy, num_rows, num_cols] = self.get_smoothness_matrix(inf_matrix.beamlets_dict)
                smoothness_X_weight = 0.6
                smoothness_Y_weight = 0.4
                obj += [obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (1 / num_rows) * cp.sum_squares(Qy @ x))]
                print(f'Adding smoothness-quadratic: {obj_funcs[i]}')
            else:
                print(f'Passing Objective: {obj_funcs[i]}')

        print('Objective done')

        print('Constraints Start')

        # constraint_def = deepcopy(clinical_criteria.get_criteria())  # get all constraints definition using clinical criteria
        constraint_def = deepcopy(opt_params_constraints)

        # Adding max/mean constraints
        dvh_constraint = []
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    struct = constraint_def[i]['parameters']['structure_name']
                    if struct != 'GTV' and struct != 'CTV':
                        if struct in my_plan.structures.get_structures():
                            if len(st.get_opt_voxels_idx(struct)) == 0:
                                print(f'Passing {struct} as no opt voxels')
                                continue
                            constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= limit / num_fractions]
                            print(f'Adding Max Dose: {constraint_def[i]}')
            elif constraint_def[i]['type'] == 'mean_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    struct = constraint_def[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    if struct in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(struct)) == 0:
                            print(f'Passing {struct} as no opt voxels')
                            continue
                        fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(struct)
                        limit = limit/fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints += [(1 / sum(st.get_opt_voxels_volume_cc(struct))) *
                                        (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(struct),
                                                             A[st.get_opt_voxels_idx(struct), :] @ x))))
                                        <= limit / num_fractions]
                        print(f'Adding Mean Dose: {constraint_def[i]}')
            elif constraint_def[i]['type'] == "dose_volume_V":
                struct = constraint_def[i]['parameters']['structure_name']
                if struct not in my_plan.structures.get_structures():
                    print(f'Passing {constraint_def[i]}, as not in struct')
                    continue 
                else:
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        print(f'Passing {struct} as no opt voxels')
                        continue
                    dvh_constraint.append(constraint_def[i])
        
        # Adding DVH constraints
        if len(dvh_constraint) > 0:
            self.add_dvh(dvh_constraint)
            print('Constraints done')
        else:
            print('No DVH constraints to add')

    def add_dvh(self, dvh_constraint: list):
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        df_dvh_constraint = pd.DataFrame()
        count = 0
        criteria = self.clinical_criteria.clinical_criteria_dict['criteria']  # for M only
        for i in range(len(dvh_constraint)):
            if 'dose_volume' in dvh_constraint[i]['type']:
                limit_key = self.matching_keys(dvh_constraint[i]['constraints'], 'limit')
                if limit_key in dvh_constraint[i]['constraints']:
                    df_dvh_constraint.at[count, 'structure_name'] = dvh_constraint[i]['parameters']['structure_name']
                    df_dvh_constraint.at[count, 'dose_gy'] = dvh_constraint[i]['parameters']['dose_gy']

                    # getting max dose_1d for the same struct_name
                    max_dose_struct = 1000
                    for j in range(len(criteria)):
                        if 'max_dose' in criteria[j]['type']:
                            if 'limit_dose_gy' in criteria[j]['constraints']:
                                org = criteria[j]['parameters']['structure_name']
                                if org == dvh_constraint[i]['parameters']['structure_name']:
                                    max_dose_struct = criteria[j]['constraints']['limit_dose_gy']
                                    max_dose_struct = self.dose_to_gy('limit_dose_gy', max_dose_struct)
                    df_dvh_constraint.at[count, 'M'] = max_dose_struct - dvh_constraint[i]['parameters']['dose_gy']

                    if 'perc' in limit_key:
                        df_dvh_constraint.at[count, 'vol_perc'] = dvh_constraint[i]['constraints'][limit_key]

                    count = count + 1

        # binary variable for dvh constraints
        b = cp.Variable( len(np.concatenate([st.get_opt_voxels_idx(org) for org in df_dvh_constraint.structure_name.to_list()])), boolean=True)

        b_start = 0
        constraints = []
        for i in range(len(df_dvh_constraint)):
            struct = df_dvh_constraint.loc[i, 'structure_name']
            limit = df_dvh_constraint.loc[i, 'dose_gy']
            v = df_dvh_constraint.loc[i, 'vol_perc']
            M = df_dvh_constraint.loc[i, 'M']

            vol_idx = st.get_opt_voxels_idx(struct)
            b_end = b_start + len(vol_idx)
            v_frac = self.my_plan.structures.get_fraction_of_vol_in_calc_box(struct)
            n_fractions = self.my_plan.get_num_of_fractions()
            vol_cc = st.get_opt_voxels_volume_cc(struct)
            total_vol_cc = sum(vol_cc)

            # voxel exceeding the dose limit will drive the binary variable to 1
            constraints += [A[vol_idx, :] @ x <= limit / n_fractions + b[b_start:b_end] * M / n_fractions]
            # the volume of voxels (in cc) exceeding the dose limit should be less than the volume (v)
            constraints += [b[b_start:b_end] @ vol_cc <= (v / v_frac) / 100 * total_vol_cc]
            b_start = b_end
            print(f'Adding DVH: {struct} limit:{limit} V:{v} M:{M}')
        self.add_constraints(constraints=constraints)

    def solve(self, return_cvxpy_prob=False, *args, **kwargs):
        """
                Return optimal solution and influence matrix associated with it in the form of dictionary
                If return_problem set to true, returns cvxpy problem instance

                :Example
                        dict = {"optimal_fluence": [..],
                        "inf_matrix": my_plan.inf_marix
                        }

                :return: solution dictionary, cvxpy problem instance(optional)
                """

        problem = cp.Problem(cp.Minimize(cp.sum(self.obj)), constraints=self.constraints)
        print('Running Optimization..')
        t = time.time()
        problem.solve(*args, **kwargs)
        elapsed = time.time() - t
        self.obj_value = problem.value
        print("Optimal value: %s" % problem.value)
        print("Elapsed time: {} seconds".format(elapsed))
        sol = {'optimal_intensity': self.vars['x'].value, 'inf_matrix': self.inf_matrix}
        if return_cvxpy_prob:
            return sol, elapsed, problem
        else:
            return sol, elapsed

class MyVisualization(pp.Visualization):
    dose_type = Literal["Absolute(Gy)", "Relative(%)"]
    volume_type = Literal["Absolute(cc)", "Relative(%)"]

    @staticmethod
    def legend_dose_storage(my_plan: Plan) -> dict:
        # dose_color = [[0.55, 0, 1], [0, 0, 1], [0, 0.5, 1], [0, 1, 0],
        #               [1, 1, 0], [1, 0.65, 0], [1, 0, 0], [0.55, 0, 0]]
        dose_color = [[0.55, 0, 0], [0, 0, 1], [0.55, 0, 1], [0, 0.5, 1], [0, 1, 0], [1, 0, 0]]
        dose_level = [0.3, 0.5, 0.7, 0.95, 1.0, 1.1]
        dose_prescription = my_plan.clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
                            my_plan.clinical_criteria.clinical_criteria_dict['num_of_fractions']
        dose_value = [item * dose_prescription for item in dose_level]
        dose_name = []
        for item in range(0, len(dose_level)):
            dose_name.append(str(round(dose_level[item] * 100, 2)) + ' % / ' +
                             str(round(dose_value[item], 3)) + ' ' + 'Gy')
        dose_storage_legend = {'dose_1d color': dose_color, 'dose_1d level': dose_level, 'dose_1d value': dose_value,
                               'dose_1d name': dose_name}
        return dose_storage_legend

    @staticmethod
    def plot_dvh(my_plan: Plan, sol: dict = None, dose_1d: np.ndarray = None, struct_names: List[str] = None,
                 dose_scale: dose_type = "Absolute(Gy)",
                 volume_scale: volume_type = "Relative(%)", **options):
        """
        Create dvh plot for the selected structures

        :param my_plan: object of class Plan
        :param sol: optimal sol dictionary
        :param dose_1d: dose_1d in 1d voxels
        :param struct_names: structures to be included in dvh plot
        :param volume_scale: volume scale on y-axis. Default= Absolute(cc). e.g. volume_scale = "Absolute(cc)" or volume_scale = "Relative(%)"
        :param dose_scale: dose_1d scale on x axis. Default= Absolute(Gy). e.g. dose_scale = "Absolute(Gy)" or dose_scale = "Relative(%)"
        :keyword style (str): line style for dvh curve. default "solid". can be "dotted", "dash-dotted".
        :keyword width (int): width of line. Default 2
        :keyword colors(list): list of colors
        :keyword legend_font_size: Set legend_font_size. default 10
        :keyword figsize: Set figure size for the plot. Default figure size (12,8)
        :keyword create_fig: Create a new figure. Default True. If False, append to the previous figure
        :keyword title: Title for the figure
        :keyword filename: Name of the file to save the figure in current directory
        :keyword show: Show the figure. Default is True. If false, next plot can be append to it
        :keyword norm_flag: Use to normalize the plan. Default is False.
        :keyword norm_volume: Use to set normalization volume. default is 90 percentile.
        :return: dvh plot for the selected structures

        :Example:
        >>> Visualization.plot_dvh(my_plan, sol=sol, struct_names=['PTV', 'ESOPHAGUS'], dose_scale='Absolute(Gy)',volume_scale="Relative(%)", show=False, create_fig=True )
        """

        if dose_1d is None:
            if 'dose_1d' not in sol:
                dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
            else:
                dose_1d = sol['dose_1d']

        if sol is None:
            sol = dict()
            sol['inf_matrix'] = my_plan.inf_matrix  # create temporary solution

        # getting options_fig:
        style = options['style'] if 'style' in options else 'solid'
        width = options['width'] if 'width' in options else None
        colors = options['colors'] if 'colors' in options else None
        legend_font_size = options['legend_font_size'] if 'legend_font_size' in options else 15
        figsize = options['figsize'] if 'figsize' in options else (12, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        show = options['show'] if 'show' in options else False
        # create_fig = options['create_fig'] if 'create_fig' in options else False
        show_criteria = options['show_criteria'] if 'show_criteria' in options else None
        ax = options['ax'] if 'ax' in options else None
        fontsize = options['fontsize'] if 'fontsize' in options else 12
        legend_loc = options["legend_loc"] if "legend_loc" in options else "upper right"
        # getting norm options
        norm_flag = options['norm_flag'] if 'norm_flag' in options else False
        norm_volume = options['norm_volume'] if 'norm_volume' in options else 90
        norm_struct = options['norm_struct'] if 'norm_struct' in options else 'PTV'

        # plt.rcParams['font.size'] = font_size
        # plt.rc('font', family='serif')
        if width is None:
            width = 3
        if colors is None:
            colors = pp.Visualization.get_colors()
        if struct_names is None:
            # orgs = []
            struct_names = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        max_vol = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [struct.upper for struct in orgs]
        pres = my_plan.get_prescription()
        legend = []

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if norm_flag:
            norm_factor = pp.Evaluation.get_dose(sol, dose_1d=dose_1d, struct=norm_struct, volume_per=norm_volume) / pres
            dose_1d = dose_1d / norm_factor
        count = 0
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in struct_names:
                continue
            if my_plan.structures.get_fraction_of_vol_in_calc_box(all_orgs[i]) == 0:  # check if the structure is within calc box
                print('Skipping Structure {} as it is not within calculation box.'.format(all_orgs[i]))
                continue
            # for dose_1d in dose_list:
            #
            x, y = pp.Evaluation.get_dvh(sol, struct=all_orgs[i], dose_1d=dose_1d)
            if dose_scale == 'Absolute(Gy)':
                max_dose = np.maximum(max_dose, x[-1])
                ax.set_xlabel('Dose (Gy)', fontsize=fontsize)
            elif dose_scale == 'Relative(%)':
                x = x / pres * 100
                max_dose = np.maximum(max_dose, x[-1])
                ax.set_xlabel('Dose ($\%$)', fontsize=fontsize)

            if volume_scale == 'Absolute(cc)':
                y = y * my_plan.structures.get_volume_cc(all_orgs[i]) / 100
                max_vol = np.maximum(max_vol, y[1] * 100)
                ax.set_ylabel('Volume (cc)', fontsize=fontsize)
            elif volume_scale == 'Relative(%)':
                max_vol = np.maximum(max_vol, y[0] * 100)
                ax.set_ylabel('Volume Fraction ($\%$)', fontsize=fontsize)
            ax.plot(x, 100 * y, linestyle=style, linewidth=width, color=colors[count], label=all_orgs[i])
            count = count + 1
            legend.append(all_orgs[i])

        if show_criteria is not None:
            for s in range(len(show_criteria)):
                if 'dose_volume' in show_criteria[s]['type']:
                    x = show_criteria[s]['parameters']['dose_gy']
                    y = show_criteria[s]['constraints']['limit_volume_perc']
                    ax.plot(x, y, marker='x', color='red', markersize=20)
        # plt.xlabel('Dose (Gy)')
        # plt.ylabel('Volume Fraction (%)')
        current_xlim = ax.get_xlim()
        final_xmax = max(current_xlim[1], max_dose * 1.1)
        ax.set_xlim(0, final_xmax)
        ax.set_ylim(0, max_vol)

        ax.grid(visible=True, which='major', color='#666666', linestyle='-')
        handles, labels = ax.get_legend_handles_labels()
        print(handles, labels)

        # Show the minor grid lines with very faint and almost transparent grey lines
        # plt.minorticks_on()
        ax.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        # if norm_flag:
        #     x = pres * np.ones_like(y)
        # else:
        if dose_scale == "Absolute(Gy)":
            x = pres * np.ones_like(y)
        else:
            x = 100 * np.ones_like(y)

        ax.plot(x, y, color='black')
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        return ax

    @staticmethod
    def comp_DVHs(my_plan: Plan, sol: dict, echo_dose_1d, **options):
        '''compare the DVHs of the echo dose and the optimized dose in sol_file_path'''
        filename = options['filename'] if 'filename' in options else None

        # DVHs
        fig, ax = plt.subplots(figsize=(8, 5))
        struct_names = ['PTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
        ax = MyVisualization.plot_dvh(my_plan, dose_1d=echo_dose_1d, struct_names=struct_names, style='solid', ax=ax, **options)
        ax = MyVisualization.plot_dvh(my_plan, sol, struct_names=struct_names, style='dotted', ax=ax, norm_flag=False, **options)


        # Collect handles and labels separately
        handles, labels = ax.get_legend_handles_labels()
        print(labels)
        handles = handles[:6]
        labels = labels[:6]
        # Change struct names to camel case and LUNGS_NOT_GTV to Lungs
        labels = [l.replace('_NOT_GTV', '') for l in labels] 
        for i, l in enumerate(labels):
            if 'GTV' not in l and 'PTV' not in l:
                labels[i] = l.capitalize()
        print(labels)

        # Create dummy lines for the legend
        solid_line = ax.plot([], [], linestyle='-', color='black', label='ECHO')[0]
        dotted_line = ax.plot([], [], linestyle=':', color='black', label='GPT-Plan')[0]
        handles.extend([solid_line, dotted_line])
        labels.extend(['ECHO', 'GPT-Plan'])
        ax.legend(handles, labels, prop={'size': options['legend_font_size']}, loc=options['legend_loc'])

        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Plot saved as {filename}")
        plt.show()

    @staticmethod
    def plot_2d_slice(my_plan: Plan = None, sol: dict = None, dose_3d : np.ndarray = None,
                      ct: pp.ct.CT = None, structs: pp.structures.Structures = None,
                      slice_num: int = 40, struct_names: List[str] = None, show_dose: bool = False,
                      show_struct: bool = True, show_isodose: bool = False,
                      **options) -> None:
        """
        Plot 2d view of ct, dose_1d, isodose and struct_name contours

        :param sol: solution to optimization
        :param my_plan: object of class Plan
        :param ct: Optional. object of class CT
        :param structs: Optional. object of class structs
        :param slice_num: slice number
        :param struct_names: structures for which contours to be displayed on the slice view. e.g. struct_names = ['PTV, ESOPHAGUS']
        :param show_dose: view dose_1d on the slice
        :param show_struct: view struct_name on the slice
        :param show_isodose: view isodose
        :param dpi: Default dpi=100 for figure
        :return: plot 2d view of ct, dose_1d, isodose and struct_name contours

        :Example:
        >>> Visualization.plot_2d_slice(my_plan, sol=sol, slice_num=50, struct_names=['PTV'], show_isodose=False)
        """

        # getting options_fig:
        figsize = options['figsize'] if 'figsize' in options else (8, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        dpi = options['dpi'] if 'dpi' in options else 100
        show = options['show'] if 'show' in options else False
        ax = options['ax'] if 'ax' in options else None
        bbox_to_anchor = options['bbox_to_anchor'] if 'bbox_to_anchor' in options else (1.5, 1)
        bbox_loc = options['bbox_loc'] if 'bbox_loc' in options else 'upper left'
        sigma = options['sigma'] if 'sigma' in options else 3
        struct_in_contour = options['struct_in_contour'] if 'struct_in_contour' in options else False
        colorbar = options['colorbar'] if 'colorbar' in options else False

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.rcParams["figure.autolayout"] = True
        if ct is None:
            ct = my_plan.ct
        ct_hu_3d = ct.ct_dict['ct_hu_3d'][0]
        ax.imshow(ct_hu_3d[slice_num, :, :], cmap='gray')

        # adjust the main plot to make room for the legends
        plt.subplots_adjust(left=0.2)
        if sol is not None or dose_3d is not None:
            show_dose = True
        if show_dose:
            if dose_3d is not None:
                dose_3d = dose_3d
            else:
                dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
                dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
                dose_3d = gaussian_filter(dose_3d, sigma=sigma)
            masked = np.ma.masked_where(dose_3d[slice_num, :, :] <= 0, dose_3d[slice_num, :, :])
            im = ax.imshow(masked, alpha=0.4, interpolation='none', cmap='rainbow')
            
            if colorbar:
                plt.colorbar(im, ax=ax, pad=0.1)

        if show_isodose:
            if not show_dose:
                dose_1d = sol['inf_matrix'].A * sol['optimal_intensity'] * my_plan.get_num_of_fractions()
                dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
            dose_legend = MyVisualization.legend_dose_storage(my_plan)
            ax.contour(dose_3d[slice_num, :, :], dose_legend['dose_1d value'], colors=dose_legend['dose_1d color'], linewidths=0.5, zorder=2)
            dose_list = []
            for item in range(0, len(dose_legend['dose_1d name'])):
                dose_list.append(Line2D([0], [0], color=dose_legend['dose_1d color'][item], lw=1, label=dose_legend['dose_1d name'][item]))
            ax.add_artist(ax.legend(handles=dose_list, bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0.))
        if title is not None:
            ax.set_title('{}: Axial View - Slice #: {}'.format(title, slice_num))
        else:
            ax.set_title('Axial View - Slice #: {}'.format(slice_num))

        if show_struct:
            if structs is None:
                structs = my_plan.structures
            if struct_names is None:
                struct_names = structs.structures_dict['name']
            struct_masks = structs.structures_dict['structure_mask_3d']
            all_mask = []
            colors = MyVisualization.get_colors()
            for i in range(len(struct_names)):
                ind = structs.structures_dict['name'].index(struct_names[i])
                cmap = mpl.colors.ListedColormap(colors[i])
                if struct_in_contour:
                    contours = measure.find_contours(struct_masks[ind][slice_num, :, :], 0.5)
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=colors[i])
                else:
                    masked_struct = np.ma.masked_where(struct_masks[ind][slice_num, :, :] == 0, struct_masks[ind][slice_num, :, :])
                    ax.imshow(masked_struct, alpha=0.4, cmap=cmap)
            labels = [struct for struct in struct_names]
            # get the colors of the values, according to the
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # rax.labels = labels
            ax.legend(handles=patches, bbox_to_anchor=(0.1, 0.8), loc=2, borderaxespad=0.)
            # bbox_transform=fig.transFigure)
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        return ax

    @staticmethod
    def box_plot(csv_path='./debug/csv_rag/comp_3plans*.csv', comp_values=('ECHO', 'GPT w/ Retrieve'), save_path='./debug/pdf/comp_echo_gpt.pdf'):
        # Increase text size
        plt.rcParams.update({'font.size': 13})

        csv_files = glob.glob(csv_path)
        print(csv_files)

        # Read all CSV files into a list of DataFrames
        dfs = [pd.read_csv(csv) for csv in csv_files]
        # Replace NaN values in 'structure_name' with 'Unknown' for HI and CI
        dfs = [df.fillna({'structure_name': ''}) for df in dfs]
        # Remove all rows where 'structure_name' contains 'RIND_x'
        dfs = [df[~df['structure_name'].fillna('').str.contains('RIND_')] for df in dfs]
        # Change the 'structure_name' to Camel case for all structure names
        dfs = [df.replace({'structure_name': {'PTV': 'PTV', 'CTV': 'CTV', 'GTV': 'GTV', 'LUNGS_NOT_GTV': 'Lungs', 'HEART': 'Heart', 'CORD': 'Cord', 'ESOPHAGUS': 'Esophagus'}}) for df in dfs]

        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Simplify constraint labels
        def simplify_constraint(constraint):
            constraint = re.sub(r'D\((\d+(\.\d+)?)%\)', r'D\1', constraint)
            constraint = re.sub(r'V\((\d+(\.\d+)?)Gy\)', r'V\1', constraint)
            return constraint

        combined_df['constraint'] = combined_df['constraint'].apply(simplify_constraint)

        # Combine 'constraint' and 'structure_name' to create unique labels for the x-axis
        combined_df['label'] = ''
        for i in range(len(combined_df)):
            if combined_df['structure_name'][i] == '':
                combined_df['label'][i] = combined_df['constraint'][i]
            else:
                combined_df['label'][i] = combined_df['structure_name'][i] + '(' + combined_df['constraint'][i] + ')'

        # Define the groups
        group1_labels = [
            'GTV(max_dose)', 'PTV(max_dose)', 'PTV(D95.0)', 'Cord(max_dose)', 
            'Lungs(mean_dose)', 'Lungs(V20.0)', 'Lungs(V5.0)', 
            'Esophagus(max_dose)', 
        ]

        # Filter the DataFrame based on the groups
        group1_df = combined_df[combined_df['label'].isin(group1_labels)]
        group2_df = combined_df[~combined_df['label'].isin(group1_labels)]

        # Prepare the data for plotting
        plot_data_group1 = pd.melt(group1_df, id_vars=['label'], value_vars=[comp_values[0], comp_values[1]], var_name='Method', value_name='Value')
        plot_data_group2 = pd.melt(group2_df, id_vars=['label'], value_vars=[comp_values[0], comp_values[1]], var_name='Method', value_name='Value')

        plot_data_group1['Method'] = plot_data_group1['Method'].replace({comp_values[0]: 'ECHO', comp_values[1]: 'GPT'})
        plot_data_group2['Method'] = plot_data_group2['Method'].replace({comp_values[0]: 'ECHO', comp_values[1]: 'GPT'})

        # Create the figure and the first set of axes
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # Define a common color palette
        palette = "Set1"

        # Plot group 1 on the left y-axis
        sns.boxplot(x='label', y='Value', hue='Method', data=plot_data_group1, ax=ax1, palette=palette, width=0.4)
        ax1.set_xlabel('Criteria')
        ax1.set_ylabel('Values (Gy or %)')

        # Create the second set of axes
        ax2 = ax1.twinx()
        # Plot group 2 on the right y-axis
        sns.boxplot(x='label', y='Value', hue='Method', data=plot_data_group2, ax=ax2, palette=palette, width=0.4)
        ax2.set_ylabel('Values (Gy or %)')
        ax2.get_legend().remove()

        # Manually set the x-tick labels to ensure they are displayed correctly
        labels = list(group1_labels) + list(group2_df['label'].unique())
        labels = [label.replace('(', '\n(') for label in labels]  # Insert newline before '('

        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=90)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=90)

        # Add a vertical line to separate the two groups
        separation_index = len(group1_labels) - 0.5
        ax1.axvline(x=separation_index, color='gray', linestyle='--')
        ax2.axvline(x=separation_index, color='gray', linestyle='--')

        # Calculate and annotate mean values for each constraint
        mean_values_group1 = group1_df.groupby('label')[comp_values].mean().reset_index()
        mean_values_group2 = group2_df.groupby('label')[comp_values].mean().reset_index()

        # Function to determine the unit based on the constraint
        def get_unit(label):
            if 'D' in label or 'dose' in label:
                return 'Gy'
            elif 'V' in label:
                return '%'
            else:
                return ''

        # Annotate means for group 1
        for i, row in mean_values_group1.iterrows():
            x_pos = labels.index(row['label'].replace('(', '\n('))
            unit = get_unit(row['label'])
            ax1.text(x_pos-0.25, row[comp_values[0]]+2, f'{row[comp_values[0]]:.2f}', color='red', ha='center', va='bottom')
            ax1.text(x_pos+0.25, row[comp_values[1]]+2, f'{row[comp_values[1]]:.2f}{unit}', color='blue', ha='center', va='bottom')

        # Annotate means for group 2
        for i, row in mean_values_group2.iterrows():
            x_pos = labels.index(row['label'].replace('(', '\n('))
            unit = get_unit(row['label'])
            ax2.text(x_pos-0.23, row[comp_values[0]]+0.1, f'{row[comp_values[0]]:.2f}', color='red', ha='center', va='bottom')
            ax2.text(x_pos+0.23, row[comp_values[1]]+0.1, f'{row[comp_values[1]]:.2f}{unit}', color='blue', ha='center', va='bottom')

        # handles, labels = ax1.get_legend_handles_labels()
        # red_proxy = mpatches.Patch(color='red', label='ECHO Mean')
        # blue_proxy = mpatches.Patch(color='blue', label='GPT Mean')
        # ax1.legend(handles=handles + [red_proxy, blue_proxy], labels=labels + ['ECHO Mean', 'GPT Mean'], title='Auto Plan Method:',  loc='upper right')

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=labels, title='Auto Plan Method:', loc='upper right')

        # Set the title and layout
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()        # Set the title and layout

    def box_plot2(csv_path='./debug/csv_rag/comp_3plans*.csv', comp_values=('ECHO', 'GPT w/ Retrieve'), save_path='./debug/pdf/comp_echo_gpt.pdf'):
        # Data loading and preprocessing (keep existing code until combined_df creation)
        plt.rcParams.update({'font.size': 13})
        csv_files = glob.glob(csv_path)
        dfs = [pd.read_csv(csv) for csv in csv_files]
        dfs = [df.fillna({'structure_name': ''}) for df in dfs]
        dfs = [df[~df['structure_name'].fillna('').str.contains('RIND_')] for df in dfs]
        dfs = [df.replace({'structure_name': {
            'PTV': 'PTV', 'CTV': 'CTV', 'GTV': 'GTV',
            'LUNGS_NOT_GTV': 'Lungs', 'HEART': 'Heart',
            'CORD': 'Cord', 'ESOPHAGUS': 'Esophagus'
        }}) for df in dfs]
        combined_df = pd.concat(dfs, ignore_index=True)
    
        # Create labels
        combined_df['label'] = combined_df.apply(lambda x: 
            x['constraint'] if x['structure_name'] == '' 
            else f"{x['structure_name']}({x['constraint']})", axis=1)
    
        # Get unique metrics
        metrics = combined_df['label'].unique()
        metrics = [m for m in metrics if not pd.isna(m)]  # Remove any NaN values
    
        # Calculate subplot layout
        n_metrics = len(metrics)
        n_cols = 5 
        n_rows = (n_metrics + n_cols - 1) // n_cols
    
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        palette = "Set1"
    
        # Plot each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_df = combined_df[combined_df['label'] == metric]
            
            # Prepare data
            plot_data = pd.melt(metric_df, 
                               value_vars=[comp_values[0], comp_values[1]], 
                               var_name='Method', 
                               value_name='Value')
            plot_data['Method'] = plot_data['Method'].replace({
                comp_values[0]: 'ECHO', 
                comp_values[1]: 'GPT-Plan'
            })
    
            # Create boxplot
            sns.boxplot(x='Method', y='Value', data=plot_data, ax=ax, 
                       palette=palette, width=0.6)
    
            # Add mean values
            means = metric_df[list(comp_values)].mean()
            unit = 'Gy' if 'dose' in metric.lower() or 'D' in metric else '%'
            if 'hi' in metric.lower() or 'ci' in metric.lower():
                unit = ''
            for i, (method, mean) in enumerate(means.items()):
                ax.text(i, mean, f'{mean:.2f}{unit}', 
                       ha='center', va='bottom')
    
            # Customize subplot
            ax.set_title(metric)
            ax.set_xlabel('')
            ax.set_ylabel('Value')
            # if idx >= len(metrics) - n_cols:  # Only show x-label for bottom row
            #     ax.set_xlabel('Method')
            # else:
            #     ax.set_xlabel('')
                
            # Only keep legend for first plot
            legend = ax.get_legend()
            if legend is not None:
                ax.get_legend().remove()

            # After plotting all metrics, use the last empty subplot for legend
        legend_ax = axes[14]  # Position for legend (15th position, index 14)
        legend_ax.clear() # Clear the subplot
        # Create dummy plots for legend
        legend_ax.plot([], [], color=sns.color_palette(palette)[0], label='ECHO', linewidth=10)
        legend_ax.plot([], [], color=sns.color_palette(palette)[1], label='GPT-Plan', linewidth=10)
        # Customize legend subplot
        legend_ax.legend(loc='center', fontsize=12, frameon=False)
        legend_ax.axis('off')  # Hide axes
    
        # Remove empty subplots
        for idx in range(len(metrics), len(axes)):
            if idx !=14: # skip legend ax
                fig.delaxes(axes[idx])
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class MyEvaluation(pp.Evaluation):
    dose_type = Literal["Absolute(Gy)", "Relative(%)"]
    volume_type = Literal["Absolute(cc)", "Relative(%)"]

    def __init__(self, pid, full_A=False, criteria_json_fn='./portpy_config_files/clinical_criteria/Default/Lung_2Gy_30Fx_wqx.json'):
        self.pid = pid
        self.criteria_dict = json.load(open(criteria_json_fn)) 
        # prepare data
        protocol_name = 'Lung_2Gy_30Fx'
        data_dir = r'/mnt/disk16T/datasets/PortPy_datasets/google_driver'

        data = pp.DataExplorer(data_dir=data_dir)
        data.patient_id = pid 
        self.cc = pp.ClinicalCriteria(data, protocol_name=protocol_name)
        self.cc.clinical_criteria_dict = self.criteria_dict

        self.ct = pp.CT(data)
        print('CT shape:', self.ct.ct_dict['ct_hu_3d'][0].shape)

        self.structs = pp.Structures(data)
        self.structs.create_opt_structures(opt_params=None, clinical_criteria=self.cc)
        print(self.structs.get_structures())
        self.beams = pp.Beams(data)
        self.inf_matrix = pp.InfluenceMatrix(ct=self.ct, structs=self.structs, beams=self.beams, is_full=False)  # only sparse matrix is created by author

        # echo dose
        dose_file_name = os.path.join(data_dir, pid, 'rt_dose_echo_imrt.dcm')  
        self.echo_dose_3d = pp.convert_dose_rt_dicom_to_portpy(ct=self.ct, dose_file_name=dose_file_name)
        self.echo_dose_1d = self.inf_matrix.dose_3d_to_1d(dose_3d=self.echo_dose_3d)

        self.my_plan = pp.Plan(self.structs, self.beams, self.inf_matrix, self.ct, self.cc)
        self.cc.clinical_criteria_dict = self.refine_clinical_criteria()

    def compare_doses_in_folder(self, dose_dirs): 
        if isinstance(dose_dirs, str):
            dose_dirs = [dose_dirs]

        files = [os.path.join(dose_dir,f) for dose_dir in dose_dirs for f in os.listdir(dose_dir) if f.endswith('.pkl')]

        # read each file and rank the doses based on the abs diff between the plan value and goal
        diff = []
        pbar = tqdm(files)
        for file in pbar:
            pbar.set_description(f"Processing {os.path.basename(file)}")
            sol = self.get_sol(file, verbose=False)
            _, _, eval_df = self.get_eval(sol=sol)
            # remove rows with 'struture_name' contain 'RIND'
            eval_df = eval_df[~eval_df['structure_name'].str.contains('RIND', na=False)]
            score, ci = calculate_score(eval_df)
            diff.append((round(score, 3), ci))

            # replace the orginal dose file with x
            if '_x' not in file:
                # get the file name without extension
                dose_dir = os.path.dirname(file)
                file_name = file.split('/')[-1].split('.')[0]
                with open(os.path.join(dose_dir, file_name+'_x.pkl'), 'wb') as f:
                    pickle.dump(sol['optimal_intensity'], f)
                os.remove(file)

        # sort the diff list by ci and then by score
        sorted_diff = sorted(list(zip(files, diff)), key=lambda x: (x[1][1], x[1][0]), reverse=True)
        for file, d in sorted_diff:
            print(file, d)
        # save the sorted_diff to a csv file
        with open(os.path.join(dose_dirs[-1], 'sorted_diff.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_diff)

    def get_sol(self, sol_file_path, verbose=True):
        if verbose:
            print('using sol_file_path {}'.format(sol_file_path))

        with open(sol_file_path, 'rb') as f:
            sol = pickle.load(f)
        if isinstance(sol, np.ndarray):
            sol = {'optimal_intensity': sol, 'inf_matrix': self.inf_matrix}
        else:
            sol['inf_matrix'] = self.inf_matrix
        return sol

    def get_eval(self, sol: dict = None, sol_file_path: str = None, save_path: str=None):
        assert sol_file_path or sol, 'sol or sol_file_path must be provided'
        if sol is None:
            sol = self.get_sol(sol_file_path)

        df = pp.Evaluation.display_clinical_criteria(self.my_plan, sol, clinical_criteria=self.cc, return_df=True).data
        # add CI and HI
        ci = self.get_conformity_index(sol)        
        hi = pp.Evaluation.get_homogeneity_index(self.my_plan, sol)
        echo_ci = self.get_conformity_index(dose_3d=self.echo_dose_3d)
        echo_hi = pp.Evaluation.get_homogeneity_index(self.my_plan, dose_3d=self.echo_dose_3d)
        new_rows = pd.DataFrame([
            {'constraint': 'CI', 'Plan Value': round(ci, 2), 'Goal': round(echo_ci, 2), 'Limit': 0.7, 'Status': ''},
            {'constraint': 'HI', 'Plan Value': round(hi, 2), 'Goal': round(echo_hi, 2), 'Limit': 0.2, 'Status': ''}
        ])
        df = pd.concat([df, new_rows], ignore_index=True)
        # insert a new col "Priority" to df
        df.insert(6, 'Priority', [2,2,2,1,4,3,4,5,5,5,5,5,7,7,7,7,6,6])

        # Apply the function to extract numeric values for comparison
        df[['Plan Value', 'Goal', 'Limit']] = df[['Plan Value', 'Goal', 'Limit']].applymap(extract_numeric)

        # determine if the plan value meets the goal/limit and format the criteria
        def process_criteria(row):
            constraint = row['constraint']
            limit = row['Limit']
            goal = row['Goal']
            value = row['Plan Value']

            # Determine if the plan value meets the goal/limit
            if constraint == 'D(95.0%)' or constraint == 'CI':
                if pd.notna(goal) and value >= goal:
                    status = f'{value} >= Goal and Limit; Met Both Goal and Limit'
                elif pd.notna(limit) and value >= limit:
                    status = f'{value} >= Limit; Met Limit but Not Goal'
                else:
                    status = f'{value} < Goal and Limit; Not Met Limit or Goal'
                limit_str = f"{constraint} ≥ {limit}"
                goal_str = f"{constraint} ≥ {goal}"
            else:
                if pd.notna(goal) and value <= goal:
                    status = f'{value} <= Goal and Limit; Met Both Goal and Limit'
                elif pd.notna(limit) and value <= limit:
                    status = f'{value} <= Limit; Met Limit but Not Goal'
                else:
                    status = f'{value} > Limit and Limit; Not Met Limit or Goal'
                limit_str = f"{constraint} ≤ {limit}"
                goal_str = f"{constraint} ≤ {goal}"

            return status, limit_str, goal_str
        
        df_display = df.copy()
        df_display[['Status', 'Limit', 'Goal']] = df_display.apply(process_criteria, axis=1, result_type='expand')
        df_display.rename(columns={'constraint': 'Constraint', 'structure_name': 'Struct', 'Limit': 'Criterion(Limit)', 'Goal': 'Criterion(Goal)', 'Plan Value': 'Achieved Value', 'Status': 'Status'}, inplace=True)
        df_display.drop(columns=['Constraint'], inplace=True)
        md = df_display.to_markdown(index=False)
        
        # save df to csv file
        if save_path:
            df.to_csv(os.path.join(save_path, f'eval_{self.pid}.csv'), index=False)
        return md, df_display, df

    def get_conformity_index(self, sol=None, dose_3d=None, target_structure='PTV', percentile=1.0) -> float:
        """
        Calculate conformity index for the dose
        Closer to 1 is more better

        :param my_plan: object of class Plan
        :param sol: optimal solution dictionary
        :param dose_3d: dose in 3d array
        :param target_structure: target structure name
        :param percentile: reference isodose
        :return: paddick conformity index

        """
        # calulating paddick conformity index
        pres = self.my_plan.get_prescription()
        if dose_3d is None:
            dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * self.my_plan.get_num_of_fractions())
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
        pres_iso_dose_mask = (dose_3d >= pres * percentile).astype(int)
        V_iso_pres = np.count_nonzero(pres_iso_dose_mask)
        ptv_mask = self.my_plan.structures.get_structure_mask_3d(target_structure)
        V_ptv = np.count_nonzero(ptv_mask)
        V_pres_iso_ptv = np.count_nonzero(pres_iso_dose_mask * ptv_mask)
        conformity_index = V_pres_iso_ptv * V_pres_iso_ptv / (V_ptv * V_iso_pres + 1e-5)
        return conformity_index

    def compare_with_echo(self, sol_file_path):
        sol = self.get_sol(sol_file_path)

        # Get the clinical criteria for the original plan
        df_my_plan = pp.Evaluation.display_clinical_criteria(self.my_plan, sol=sol, return_df=True).data

        # Get the clinical criteria for the echo plan
        df_echo_plan = pp.Evaluation.display_clinical_criteria(self.my_plan, dose_1d=self.echo_dose_1d, return_df=True).data

        # Apply the function to extract numeric values for comparison
        df_my_plan['Plan Value'] = df_my_plan['Plan Value'].apply(extract_numeric)
        df_echo_plan['Plan Value'] = df_echo_plan['Plan Value'].apply(extract_numeric)

        # Rename columns for clarity
        df_my_plan.rename(columns={'Plan Value': 'myPlan Value'}, inplace=True)
        df_echo_plan.rename(columns={'Plan Value': 'echoPlan Value'}, inplace=True)

        # Merge the two DataFrames on the constraint column
        df_comparison = pd.merge(df_my_plan[['constraint', 'structure_name', 'Limit', 'Goal', 'myPlan Value']],
                                df_echo_plan[['constraint', 'structure_name','Limit', 'Goal', 'echoPlan Value']],
                                on=['constraint', 'structure_name', 'Limit', 'Goal'], how='left')

        # Add CI and HI values
        ci_my_plan = self.get_conformity_index(sol)
        hi_my_plan = pp.Evaluation.get_homogeneity_index(self.my_plan, sol)
        ci_echo_plan = self.get_conformity_index(dose_3d=self.echo_dose_3d)
        hi_echo_plan = pp.Evaluation.get_homogeneity_index(self.my_plan, dose_3d=self.echo_dose_3d)

        # Create new rows for CI and HI
        new_rows = pd.DataFrame([
            {'constraint': 'CI', 'myPlan Value': round(ci_my_plan, 2), 'echoPlan Value': round(ci_echo_plan, 2)},
            {'constraint': 'HI', 'myPlan Value': round(hi_my_plan, 2), 'echoPlan Value': round(hi_echo_plan, 2)}
        ])

        # Concatenate the new rows to the existing DataFrame
        df_comparison = pd.concat([df_comparison, new_rows], ignore_index=True)

        # Convert df_comparison table to markdown table
        md = df_comparison.to_markdown(index=False)
        df_comparison.to_csv(f'debug/comp_{self.pid}.csv', index=False)
        print(f'Saved comparison table to debug/comp_{self.pid}.csv')
        
        return md, df_comparison
    
    def compare_echo_gpt_gptRAG(self, gpt_sol_file_path, gptRAG_sol_file_path, save_path):
        sol_gpt = self.get_sol(gpt_sol_file_path)
        sol_rag = self.get_sol(gptRAG_sol_file_path)

        # Get the clinical criteria 
        df1 = pp.Evaluation.display_clinical_criteria(self.my_plan, sol=sol_gpt, return_df=True).data
        df2 = pp.Evaluation.display_clinical_criteria(self.my_plan, sol=sol_rag, return_df=True).data
        df3 = pp.Evaluation.display_clinical_criteria(self.my_plan, dose_1d=self.echo_dose_1d, return_df=True).data

        # Apply the function to extract numeric values for comparison
        df1['Plan Value'] = df1['Plan Value'].apply(extract_numeric)
        df2['Plan Value'] = df2['Plan Value'].apply(extract_numeric)
        df3['Plan Value'] = df3['Plan Value'].apply(extract_numeric)

        # Rename columns for clarity
        df1.rename(columns={'Plan Value': 'GPT w/o Retrieve'}, inplace=True)
        df2.rename(columns={'Plan Value': 'GPT w/ Retrieve'},  inplace=True)
        df3.rename(columns={'Plan Value': 'ECHO'}, inplace=True)

        # Merge the first two DataFrames on the constraint and structure_name columns
        df_temp = pd.merge(df1[['constraint', 'structure_name', 'GPT w/o Retrieve']],
                           df2[['constraint', 'structure_name', 'GPT w/ Retrieve']],
                           on=['constraint', 'structure_name'], how='left')
        
        # Merge the result with the third DataFrame
        df_comparison = pd.merge(df_temp,
                                 df3[['constraint', 'structure_name', 'ECHO']],
                                 on=['constraint', 'structure_name'], how='left')

        # Add CI and HI values
        ci1 = self.get_conformity_index(sol_gpt)
        hi1 = pp.Evaluation.get_homogeneity_index(self.my_plan, sol_gpt)
        ci2 = self.get_conformity_index(sol_rag)
        hi2 = pp.Evaluation.get_homogeneity_index(self.my_plan, sol_rag)
        ci3 = self.get_conformity_index(dose_3d=self.echo_dose_3d)
        hi3 = pp.Evaluation.get_homogeneity_index(self.my_plan, dose_3d=self.echo_dose_3d)

        # Create new rows for CI and HI
        new_rows = pd.DataFrame([
            {'constraint': 'CI', 'GPT w/o Retrieve': round(ci1, 2), 'GPT w/ Retrieve': round(ci2, 2), 'ECHO': round(ci3, 2)},
            {'constraint': 'HI', 'GPT w/o Retrieve': round(hi1, 2), 'GPT w/ Retrieve': round(hi2, 2), 'ECHO': round(hi3, 2)}
        ])

        # Concatenate the new rows to the existing DataFrame
        df_comparison = pd.concat([df_comparison, new_rows], ignore_index=True)

        # Convert df_comparison table to markdown table
        md = df_comparison.to_markdown(index=False)
        save_path = os.path.join(save_path, f'comp_3plans_{self.pid}.csv')
        df_comparison.to_csv(save_path, index=False)
        print(f'Saved comparison table to {save_path}')
        
        return md, df_comparison

    @staticmethod
    def compare_all_patients_3plans(csv_files='./debug/csv_rag/*3plans*csv'):
        # Read all CSV files into a list of DataFrames
        csv_files = glob.glob(csv_files)
        dfs = [pd.read_csv(csv) for csv in csv_files]
        # Replace NaN values in 'structure_name' with 'Unknown' for HI and CI
        dfs = [df.fillna({'structure_name': ''}) for df in dfs]
        # Remove all rows where 'structure_name' contains 'RIND_x'
        dfs = [df[~df['structure_name'].fillna('').str.contains('RIND_')] for df in dfs]
    
        # Combine all DataFrames into a single DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)
    
        # Simplify constraint labels
        def simplify_constraint(constraint):
            constraint = re.sub(r'D\((\d+(\.\d+)?)%\)', r'D\1', constraint)
            constraint = re.sub(r'V\((\d+(\.\d+)?)Gy\)', r'V\1', constraint)
            return constraint
    
        combined_df['constraint'] = combined_df['constraint'].apply(simplify_constraint)
    
        # Combine 'constraint' and 'structure_name' to create unique labels for the x-axis
        combined_df['label'] = ''
        for i in range(len(combined_df)):
            if combined_df['structure_name'][i] == '':
                combined_df['label'][i] = combined_df['constraint'][i]
            else:
                combined_df['label'][i] = combined_df['structure_name'][i] + '(' + combined_df['constraint'][i] + ')'
    
        # Calculate mean and std values for each label
        mean = combined_df.groupby('label').mean()
        std = combined_df.groupby('label').std()
    
        # p-value calculation
        p_values = {}
        for label in combined_df['label'].unique():
            gpt_wo_retrieve = combined_df[combined_df['label'] == label]['GPT w/o Retrieve']
            gpt_w_retrieve = combined_df[combined_df['label'] == label]['GPT w/ Retrieve']
            echo = combined_df[combined_df['label'] == label]['ECHO']
            
            # Perform paired t-tests
            t_stat_1, p_value_1 = stats.ttest_rel(gpt_w_retrieve, gpt_wo_retrieve)
            t_stat_2, p_value_2 = stats.ttest_rel(gpt_w_retrieve, echo)
            
            # Store p-values in a dictionary
            p_values[label] = {
                'GPT w/ Retrieve vs GPT w/o Retrieve': p_value_1,
                'GPT w/ Retrieve vs ECHO': p_value_2
            }
    
        # Create a new DataFrame with mean and std values
        df = pd.concat([mean, std], axis=1)
        df.columns = pd.MultiIndex.from_product([['Mean', 'Std'], ['GPT w/o Retrieve', 'GPT w/ Retrieve', 'ECHO']])
        df = df.reset_index()
    
        # Add p-values to the DataFrame
        p_values_df = pd.DataFrame.from_dict(p_values, orient='index')
        p_values_df.columns = ['p-value (GPT w/ Retrieve vs GPT w/o Retrieve)', 'p-value (GPT w/ Retrieve vs ECHO)']
        df = df.merge(p_values_df, left_on='label', right_index=True)
    
        return df

    def get_echo_eval(self):
        '''use echo plan to refine the clinical criteria'''
        df = pp.Evaluation.display_clinical_criteria(self.my_plan, dose_1d=self.echo_dose_1d, return_df=True).data
        df.replace('', np.nan, inplace=True)
        # print(df)

        # Function to compare and replace Goal values
        def compare_and_replace(row):
            goal = row['Goal']
            echo = row['Plan Value']

            # Function to strip units and convert to numeric
            def to_numeric(value):
                if pd.isna(value):
                    return np.nan
                # Remove units like Gy and %
                value = re.sub(r'[^\d.]+', '', str(value))
                return pd.to_numeric(value, errors='coerce')

            numeric_goal = to_numeric(goal)
            numeric_echo = to_numeric(echo)

            if 'D(95' not in row['constraint']: 
                if pd.isna(numeric_goal) or (numeric_goal > numeric_echo):
                    print(f"Replaced Goal value for {row['structure_name']} - {row['constraint']} with {echo}")
                    row['Goal'] = numeric_echo 
                else:
                    row['Goal'] = numeric_goal
            else:
                row['Goal'] = numeric_goal
            return row

        # Apply the function to each row
        df = df.apply(compare_and_replace, axis=1)
        # print(df)
        return df

    def refine_clinical_criteria(self):
        df = self.get_echo_eval()
        cc = self.criteria_dict

        for c in cc['criteria']:
            name = c['parameters'].get('structure_name')
            cst_type = c['type']
            volume_perc = c['parameters'].get('volume_perc')
            dose_gy = c['parameters'].get('dose_gy')

            if cst_type == 'dose_volume_D':
                col_constraint = f'D({volume_perc:.1f}%)'
            elif cst_type == 'dose_volume_V':
                col_constraint = f'V({dose_gy:.1f}Gy)'
            else:
                col_constraint = cst_type

            row = df[(df['structure_name'] == name) & (df['constraint'] == col_constraint)]
            if not row.empty:
                goal_value = row.iloc[0]['Goal']
                if 'limit_dose_gy' in c['constraints'] or 'goal_dose_gy' in c['constraints']:
                    c['constraints']['goal_dose_gy'] = goal_value
                elif 'limit_volume_perc' in c['constraints'] or 'goal_volume_perc' in c['constraints']:
                    c['constraints']['goal_volume_perc'] = goal_value
        return cc

class IMRT:
    ''' The IMRT (Intensity-Modulated Radiation Therapy) class optimizes a radiation treatment plan for a patient.  It contains information about the patient, the radiation beams, and the treatment protocol.  '''
    def __init__(self, 
                 pid: str,
                 criteria_json_fn: str = "./portpy_config_files/clinical_criteria/Default/Lung_2Gy_30Fx_wqx.json",
                 init_optim_param_json_file: str = "./portpy_config_files/optimization_params/optimization_params_Lung_2Gy_30Fx_congliu.json",
                 data_dir: str = r'/mnt/disk16T/datasets/PortPy_datasets/google_driver',
                 beam_ids: List[str] = None,
                 protocol_name: str = 'Lung_2Gy_30Fx',
                 time_limit: int = 1200
                 ) -> None:

        self.pid = pid
        self.data_dir = data_dir
        self.beam_ids = beam_ids
        self.protocol_name = protocol_name
        self.solution = None
        self.criteria_json_fn = criteria_json_fn
        self.criteria_dict = json.load(open(criteria_json_fn)) 
        self.time_limit = time_limit
        self.init_optim_param_json_file = init_optim_param_json_file
        self.load_data()

    def load_data(self):
        # load data
        data = pp.DataExplorer(data_dir=self.data_dir)
        data.patient_id = self.pid

        self.ct = pp.CT(data)
        self.structs = pp.Structures(data)
        self.beams = pp.Beams(data)

        self.cc = pp.ClinicalCriteria(data, self.protocol_name)
        self.cc.clinical_criteria_dict = self.criteria_dict

        # Creating optimization structures (i.e., Rinds) 
        with open(self.init_optim_param_json_file, 'r') as f:
            op_dict = json.load(f)
        self.structs.create_opt_structures(opt_params=op_dict, clinical_criteria=self.cc)
        print(self.structs.get_structures())

        # Loading sparse influence matrix
        self.inf_matrix = pp.InfluenceMatrix(ct=self.ct, structs=self.structs, beams=self.beams, is_full=False)

    def do_optim(self, optim_param_json):
        # Loading hyper-parameter values for optimization problem
        op_dict = json.loads(optim_param_json)

        # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
        self.plan = pp.Plan(ct=self.ct, structs=self.structs, beams=self.beams, inf_matrix=self.inf_matrix, clinical_criteria=self.cc)

        # create cvxpy problem using the clinical criteria and optimization parameters
        pre_vars = None
        if self.solution:
            pre_value = self.solution['optimal_intensity']  # numpy array
            pre_vars = {'x': cp.Variable(len(pre_value), value=pre_value, pos=True, name='x')}  # creating variable for beamlet intensity
        optim = MyOptimization(self.plan, opt_params=op_dict, vars=pre_vars)
        # add objectives from optim_param: quadratic-overdose, quadratic-underdose, quadratic, smoothness-quadratic
        # add constrains from optim_param: Ring max dose constraints
        # add max and mean dose constrains from criteria
        optim.create_cvxpy_problem()
        # add dvh constrain from criteria

        # run optimization using cvxpy and one of the supported solvers and save the optimal solution in optimized_plan 
        # sol, elapsed_time = optim.solve(solver='GUROBI', verbose=True, TimeLimit=self.time_limit)  # MOSEK, GUROBI, in seconds 20 mins
        sol, elapsed_time = optim.solve(solver='GUROBI', verbose=True, TimeLimit=self.time_limit)
        self.solution = sol
        print('Optim Done!')

        # evaluate 
        dose_md, dose_df = self.evaluate_plan(self.plan, sol)
        return sol, dose_md, dose_df, elapsed_time

    def evaluate_plan(self, plan: pp.Plan, solution: dict) -> str:
        eval = MyEvaluation(self.pid, self.criteria_json_fn)
        md, df_dis, _ = eval.get_eval(solution)
        return md, df_dis 
        
def test_JSON2Markdown():
  # Specify the paths to the JSON files
  optim_param_json_fn = "/home/congliu/linatech/portpy/PortPy20240624/PortPy/portpy/config_files/optimization_params/optimization_params_Lung_2Gy_30Fx.json"
  criteria_json_fn = "/home/congliu/linatech/portpy/PortPy20240624/PortPy/portpy/config_files/clinical_criteria/Default/Lung_2Gy_30Fx.json"

  # Create an instance of JSON2Markdown
  converter = JSON2Markdown(criteria_json_fn, optim_param_json_fn)

  # Convert JSON to Markdown
  markdown = converter.json_to_markdown()
  print(markdown)

  # Convert Markdown to JSON
  new_markdown_table = '| ROI Name      | Objective Type       | Target Gy            | % Volume   | Weight   |\n|:--------------|:---------------------|:---------------------|:-----------|:---------|\n| PTV           | quadratic-overdose   | prescription_gy      | NA         | 10000    |\n| PTV           | quadratic-underdose  | prescription_gy      | NA         | 100000   |\n| PTV           | max_dose             | 69                   | NA         | NA       |\n| GTV           | max_dose             | 69                   | NA         | NA       |\n| CORD          | linear-overdose      | 50                   | NA         | 100      |\n| CORD          | quadratic            | NA                   | NA         | 10       |\n| CORD          | max_dose             | 50                   | NA         | NA       |\n| ESOPHAGUS     | quadratic            | NA                   | NA         | 20       |\n| ESOPHAGUS     | max_dose             | 66                   | NA         | NA       |\n| ESOPHAGUS     | mean_dose            | 34                   | NA         | NA       |\n| ESOPHAGUS     | dose_volume_V        | 60                   | 17         | NA       |\n| HEART         | quadratic            | NA                   | NA         | 20       |\n| HEART         | max_dose             | 66                   | NA         | NA       |\n| HEART         | mean_dose            | 27                   | NA         | NA       |\n| HEART         | dose_volume_V        | 30                   | NA         | NA       |\n| HEART         | dose_volume_V        | 30                   | 50         | NA       |\n| LUNGS_NOT_GTV | quadratic            | NA                   | NA         | 10       |\n| LUNGS_NOT_GTV | max_dose             | 66                   | NA         | NA       |\n| LUNGS_NOT_GTV | mean_dose            | 21                   | NA         | NA       |\n| LUNGS_NOT_GTV | dose_volume_V        | 20                   | 37         | NA       |\n| LUNG_L        | quadratic            | NA                   | NA         | 10       |\n| LUNG_L        | max_dose             | 66                   | NA         | NA       |\n| LUNG_R        | quadratic            | NA                   | NA         | 10       |\n| LUNG_R        | max_dose             | 66                   | NA         | NA       |\n| RIND_0        | quadratic            | NA                   | NA         | 5        |\n| RIND_0        | max_dose             | 1.1*prescription_gy  | NA         | NA       |\n| RIND_1        | quadratic            | NA                   | NA         | 5        |\n| RIND_1        | max_dose             | 1.05*prescription_gy | NA         | NA       |\n| RIND_2        | quadratic            | NA                   | NA         | 3        |\n| RIND_2        | max_dose             | 0.9*prescription_gy  | NA         | NA       |\n| RIND_3        | quadratic            | NA                   | NA         | 3        |\n| RIND_3        | max_dose             | 0.85*prescription_gy | NA         | NA       |\n| RIND_4        | quadratic            | NA                   | NA         | 3        |\n| RIND_4        | max_dose             | 0.75*prescription_gy | NA         | NA       |\n| SKIN          | max_dose             | 60                   | NA         | NA       |\n| NA            | smoothness-quadratic | NA                   | NA         | 100      |'

  criteria_json, optim_param_json = converter.markdown_to_json(new_markdown_table)
  print(converter.json_to_markdown(criteria_json, optim_param_json))

def test_all():
    root_path = Path('debug/portpy_config_files/')
    optim_param_json_fn = root_path / Path("optimization_params/optimization_params_Lung_2Gy_30Fx_congliu.json")
    criteria_json_fn = root_path / Path("clinical_criteria/Default/Lung_2Gy_30Fx.json")

    # Create an instance of JSON2Markdown
    # j2m = JSON2Markdown(optim_param_json_fn)
    j2m = JSON2Markdown('./debug/portpy_config_files/optimization_params/optimization_params_Lung_2Gy_30Fx_congliu.json') 
    op_json = j2m.markdown_to_json(optPara_md)

    # Convert JSON to Markdown
    _markdown = j2m.json_to_markdown()
    print(_markdown)

    _optim_param = j2m.markdown_to_json(_markdown)
    print(json.dumps(json.loads(_optim_param), indent=4))

    pid = 'Lung_Phantom_Patient_1'  # 'Lung_Patient_44'
    imrt = IMRT(patient_id=pid, criteria_json_fn=criteria_json_fn)
    imrt.do_optim(_optim_param)

def test_eval(): 
    eval30 = MyEvaluation(pid='Lung_Patient_30')
    sol_file_path = ('/mnt/disk16T/datasets/PortPy_datasets/traj_files/Lung_Patient_30_07152239/solution_33_x.pkl')
    eval30.get_eval(sol_file_path=sol_file_path)

def test_comp_folder():
    eval30 = MyEvaluation(pid='Lung_Patient_30')
    dose_dirs = ['/mnt/disk16T/datasets/PortPy_datasets/traj_files_rag/Lung_Patient_30']
    eval30.compare_doses_in_folder(dose_dirs)

def test_comp_3plans():
    root_dir = '/mnt/disk16T/datasets/PortPy_datasets/'
    eval30 = MyEvaluation(pid='Lung_Patient_30')
    gpt_sol_file_path = (root_dir+'traj_files/Lung_Patient_30_0730/solution_50_x.pkl')
    gptRAG_sol_file_path = (root_dir+'traj_files_rag/Lung_Patient_30/solution_x_5.pkl')
    eval30.compare_echo_gpt_gptRAG(gpt_sol_file_path, gptRAG_sol_file_path)

def test_boxplot():
    MyVisualization.box_plot2(save_path='./debug/pdf/comp_echo_gptRAG2.pdf')

def test_comp_all_pat_3Plans():
    MyEvaluation.compare_all_patients_3plans()

if __name__ == "__main__":
    test_all()
    # test_eval()
    # test_comp_folder()
    # test_comp_3plans()
    # test_boxplot()
    # test_comp_all_pat_3Plans()