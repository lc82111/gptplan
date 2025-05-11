import io
import json
import asyncio, os, copy
import inspect
import logging
import pickle
import re
import textwrap
import time
import pandas as pd
from termcolor import colored
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn
from autogen.agentchat import Agent, ConversableAgent 
from autogen.oai.client import OpenAIWrapper
from autogen.token_count_utils import count_token, get_max_token_limit, num_tokens_from_functions
# from autogen.agentchat.contrib.capabilities.text_compressors import LLMLingua
# from autogen.agentchat.contrib.capabilities.transforms import TextMessageCompressor
import google.generativeai as genai
from prompts import get_prompt
from use_portpy import IMRT, JSON2Markdown 

# llm_lingua = LLMLingua(dict(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map="cuda:0"))
# text_compressor = TextMessageCompressor(text_compressor=llm_lingua, compression_params={"target_token":100, "keep_first_sentence":1}, cache=None)

# genai.configure(api_key='AIzaSyABK3x5mMVuZrqS9HIUgn13rmVXgZYDm0E')
# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)
# gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

def md2df(md_table):
    if 'OptPara' in md_table:
        # remove the first line 
        md_table = '\n'.join(md_table.split('\n')[1:])

    # Use pandas to read the Markdown table
    df = pd.read_table(io.StringIO(md_table), sep='|', skipinitialspace=True)
    # Clean up the column names and drop any empty columns/rows
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    # Remove the separator row (usually the second row in a markdown table)
    df = df.drop(df.index[df.iloc[:, 0].str.contains('-{3,}', na=False)]).reset_index(drop=True)
    # Remove any remaining rows that are all NaN or contain only separators
    df = df[~df.apply(lambda row: row.astype(str).str.contains('-{3,}').all() or row.isna().all(), axis=1)]
    # Reset the index after dropping rows
    df = df.reset_index(drop=True)
    return df

def check_no_errors(text):
    pattern = r'(?i)no\s+(?:\w+\s+){0,5}errors'
    match = re.search(pattern, text, re.IGNORECASE)
    return bool(match)

def check_OptPara(text):
    pattern = r'(?m)^(?=.*\|.*ROI Name.*\|)(?:.*\|.*\n){5,}'
    return bool(re.search(pattern, text))

def extract_OptPara(iter_count, response: str):
    # extract the OptPara part from the modified response
    # OptPara = re.search(r'\|.*?ROI Name.*?\|.*?(\n\|.*?\|.*?\|.*?\|.*?\|.*?\|)+', response, re.DOTALL)
    # OptPara = re.search(r'\|.*?ROI Name.*?(?:\n\|.*+)+', response, re.DOTALL)
    # OptPara = re.search(r'\|\s*ROI Name.*?\|\s*Weight\s*\|\n(?:\|(?:[^|]*\|){4,}\n)+(?=\n[^|]|\Z)',response, re.MULTILINE|re.DOTALL)
    # if OptPara is not None:
    #     OptPara = OptPara.group()
    # else:
    #     raise ValueError("OptPara not found in the dosimetrist's response.")

    lines = response.split('\n')
    table_start = None
    table_end = None

    # lines = [line for line in lines if ':---' not in line.strip()]
    lines = [line for line in lines if '```' not in line.strip()]

    for i, line in enumerate(lines):
        if '|' in line and 'ROI Name' in line:
            table_start = i
        elif table_start is not None:
            if line.strip() == '':  # If we've reached an empty line
                table_end = i
                break
            # if reached an line that doesn't have a '|' character, then it's end of the table
            elif '|' not in line.strip():  
                table_end = i
                break
            elif i == len(lines) - 1:  # If we've reached the last line
                table_end = i + 1  # Include the last line
                break
   
    if table_start is not None and table_end is not None:
        OptPara = '\n'.join(lines[table_start:table_end])
    else:
        raise ValueError("OptPara not found in the dosimetrist's response.")

    final_response = f'### OptPara Iter-{iter_count}\n' + OptPara

    # print the rest of the response after removing the OptPara part
    # rest_of_response = response.replace(OptPara, '') 
    # print(rest_of_response)

    return final_response

def add_iter_to_OptPara(iter_count, response):
    lines = response.split('\n')

    # Remove any existing iteration numbers 
    lines = [line for line in lines if '### OptPara Iter' not in line]  

    # Add the iteration number to the OptPara table
    for i, line in enumerate(lines):
        if 'ROI Name' in line and '|' in line:
            # insert the iteration number to the previouse line
            lines.insert(i, f'### OptPara Iter-{iter_count}') 
            break
            
    return '\n'.join(lines)

def parse_traj(traj_file_path, iter_idx, role):
    """
    Parses the trajectory file to return either the OptPara table or the Dosimetric Outcomes table
    based on the specified role ('TPS' for OptPara table, 'dosimetric' for Dosimetric Outcomes table) at specific iter_index
    
    Parameters:
    - traj_file_path: Path to the trajectory file.
    - iter_index: Iteration index to identify the specific iteration's data.
    - role: 'TPS' to return the OptPara table, 'dosimetric' to return the Dosimetric Outcomes table.
    
    Returns:
    - the parsed md string containing the requested table.
    """
    # Read the markdown file
    with open(traj_file_path, 'r') as file:
        content = file.read()
    
    # Define the section start based on iter_index and role
    if role.lower() == 'dosimetric':
        section_start = f"### OptPara Iter-{iter_idx}"
    elif role.lower() == 'tps':
        section_start = f"Given OptPara Iter-{iter_idx}"
    else:
        raise ValueError("Invalid role specified. Choose either 'TPS' or 'dosimetric'.")
    
    # Find the start of the section
    start_index = content.find(section_start)
    if start_index == -1:
        print("Section not found.")
        return None
    
    # Extract the section until the next section starts or file ends
    end_index = content.find("\n\n", start_index + len(section_start))
    if end_index == -1:  # This means we are at the last section
        end_index = len(content)
    
    # Extract and return the section
    return content[start_index:end_index].strip()

def simplify_messages(self, sender, messages):
    '''for dosimetrist and physicist'''
    if sender.iter_count>1 and (sender.iter_count-1)%sender.everyN==0:  # every 10 iterations
        for msg in messages[1:]:  # skip first message which is group introduction msg and without name key
            speaker = msg.get('name', self._name)
            if any(role in speaker for role in ['TPS_proxy', 'dosimetrist', 'physicist', 'human_supervisor']):
                messages.remove(msg)

        assert sender.traj != '', "Trajectory Summary is empty."
        messages.append({'role': 'user', 'name': 'optim_trajectoy_reviwer', 'content':sender.traj})


def dosimetrist_preprocess(self, sender, messages):
    if sender.iter_count==1 or sender.is_skip_physicist:
        return ''

    # 1. suggest ajustment to dosimetrist
    dose_eval = next((msg for msg in reversed(messages) if msg.get('name') == 'physicist'), None)
    assert dose_eval is not None, "Physicist's evaluation not found in the chat history."
    dose_eval = dose_eval['content']
    assert 'Optimization Priorities' in dose_eval, "Optimization Priorities not found in the physicist's evaluation." 
    dose_eval = dose_eval.split('Optimization Priorities')[1]     

    msg_to_send  = [{'role':'user', 'name':'dosimetrist', 'content':messages[-4]['content']}]  # 1. dosimetrist's message
    msg_to_send += [messages[-3]] # 2. TPS
    msg_to_send += [{'role': 'user', 'name': 'principal_physicist', 'content': dose_eval}]  # 3. physicist's evaluation
    msg_to_send += [messages[-1]]  # 4. human supervisor's feedback

    for i in range(3):
        suggestedAdjustment = self.agent_SuggestDosimetrist.generate_reply(messages=msg_to_send)
        suggestedAdjustment = suggestedAdjustment['content'] if isinstance(suggestedAdjustment, dict) else suggestedAdjustment
        break

        # self reflects and corrects the suggested adjustment
        msg_to_send.append({'role': 'user', 'content': copy.deepcopy(suggestedAdjustment)})
        critique = self.agent_SuggestDosimetristCheck.generate_reply(messages=msg_to_send)
        if isinstance(critique, dict):
            critique = critique['content']
        if 'no errors' in critique.lower():
            break
        else:
            msg_to_send.append({'role': 'user', 'content': critique})
    
    extra_msg = """\
Note:
- The "All Possible OptPara Adjustments" list all possible adjustments for each requirement, the dosimetrist does not need to use all of them. The dosimetrist should select the most appropriate adjustments based on their own judgment.
- If D95 < 60 Gy, avoid increasing PTV quadratic-overdose Weight and PTV quadratic-underdose Weight simultaneously to prevent compromising the PTV coverage.
- If D95 < 60 Gy, never decrease PTV quadratic-overdose Target Gy because it will punish the dose points above the target dose.
- If Max Dose for a struct is not met the goal, decreasing the max_dose "Target Gy" for the struct is a VERY EFFECTIVE WAY to reduce the max dose. 
"""
    suggestedAdjustment += f'\n\n{extra_msg}'
    return suggestedAdjustment

def generate_oai_reply_with_process_dosimetrist(
    self: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    """Generate a reply using autogen.oai."""
    client = self.client if config is None else config
    if client is None:
        return False, None
    if messages is None:
        messages = self._oai_messages[sender]  # sender always is the group manager

    assert self._name == "dosimetrist"

    # try cache
    if not sender.traj_file is None:
        OptPara = parse_traj(sender.traj_file, sender.iter_count, 'dosimetric')
        if OptPara is not None:
            simplify_messages(self, sender, messages)
            return True, OptPara

    premsg = dosimetrist_preprocess(self, sender, messages)
    print(colored(premsg, 'cyan'))
    msg_to_send = messages if premsg == '' else messages + [{'role': 'user', 'content': premsg}]
    response = self._generate_oai_reply_from_client(client, self._oai_system_message+msg_to_send,  self.client_cache)
    response = response['content'] if isinstance(response, dict) else response
    response = dosimetrist_post_process(self, sender.iter_count, messages, response, client)
    response = add_iter_to_OptPara(sender.iter_count, response)
    # OptPara = dosimetrist_extract_OptPara(sender.iter_count, response)

    # simplify_physicist_msgs(self, messages, -2)  # keep only takeaways in dosimetrist's chat history after reply
    simplify_messages(self, sender, messages)
    return True, response 

def dosimetrist_post_process(
    self: ConversableAgent,
    cur_iter: int,
    messages: Optional[List[Dict]] = None,
    response: str = None,
    client: Optional[OpenAIWrapper] = None,
) -> Union[str, Dict, None]:
    '''Dosimetrist self-reflection and correction'''

    # refine the OptPara with critique for maximum 6 times 
    for i in range(6): 
        print(colored(f'dosimetrist response-{i}', 'red'))
        print(colored(response, 'blue'))

        error = False
        try:
            # check the exsiting of OptPara
            if not check_OptPara(response):
                raise ValueError(f"OptPara Iter-{cur_iter} not found in your response. You are required to provide OptPara markdown table in your response.") 
                
            # multiple OptPara fractions in the response
            # if response.count('ROI Name') > 1:
            #     raise ValueError(f"Multiple OptPara fractions found in the response. Please provide only one complete OptPara Iter-{cur_iter}.")
            
            # ensure all ROIs and Type are valid
            valid_ROIs =  ['GTV', 'PTV', 'LUNGS_NOT_GTV', 'LUNG_L', 'LUNG_R', 'ESOPHAGUS', 'HEART', 'CORD', 'SKIN', 'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4', 'NA']
            valid_objs = [ 'quadratic-overdose', 'quadratic-underdose', 'quadratic', 'linear-overdose', 'smoothness-quadratic', 'max_dose', 'mean_dose', 'dose_volume_V' ]
            OptPara_md = extract_OptPara('tmp', response)
            df = md2df(OptPara_md)
            cur_ROIs = [s.strip() for s in df['ROI Name'].tolist()]
            cur_types = [s.strip() for s in df['Objective Type'].tolist()]
            n_ptv_quad_overdose = sum([1 for i, row in df.iterrows() if 'PTV' in row['ROI Name'] and 'quadratic-overdose' in row['Objective Type']])
            n_ptv_quad_underdose = sum([1 for i, row in df.iterrows() if 'PTV' in row['ROI Name'] and 'quadratic-underdose' in row['Objective Type']])

            # find valid ROIs that are not in the ROIs
            not_found_ROIs = [s for s in valid_ROIs[0:-1] if s not in cur_ROIs]  # exclude 'NA'
            if len(not_found_ROIs) > 0:
                raise ValueError(f"The following ROI names are missing in your response: {not_found_ROIs}. You should include all these ROIs in your OptPara table.") 

            # ensure 'smoothness-quadratic' is in the OptPara
            if 'smoothness-quadratic' not in cur_types:
                raise ValueError(f"The 'smoothness-quadratic' Objective Type is missing in your response. You should include it in your OptPara table. Note that the ROI Name should be 'NA' for this Objective Type.")

            # find OARs/types are not in valid_OARs/objs
            invalid_ROIs = [s for s in cur_ROIs if s not in valid_ROIs]
            if len(invalid_ROIs) > 0:
                raise ValueError(f"The following ROI names are invalid: {invalid_ROIs}. The available ROIs are: {valid_ROIs}.")

            invalid_types = [s for s in cur_types if s not in valid_objs]
            if len(invalid_types) > 0:
                raise ValueError(f"The following Objective Types are invalid: {invalid_types}. The available Objective Types are: {valid_objs}.")

            # weight and volume parameter errors
            for i, row in df.iterrows():
                struct = row['ROI Name'].strip()
                obj_type = row['Objective Type'].strip()
                target_gy = float(row['Target Gy'].strip()) if 'NA' not in row['Target Gy'] else 0
                vol = row['% Volume'].strip()
                weight = row['Weight'].strip()

                if 'PTV' in struct and 'quadratic-overdose'in obj_type and target_gy < 60 and n_ptv_quad_overdose == 1:
                    raise ValueError(f"The PTV quadratic-overdose have a Target Gy less than 60 Gy, which will punish the dose points above {target_gy} Gy. Please adjust the Target Gy to be greater than 60 Gy or give a rationale for the lower Target Gy.")

                if 'PTV' in struct and 'quadratic-underdose'in obj_type and target_gy < 60 and n_ptv_quad_underdose == 1:
                    raise ValueError(f"The PTV quadratic-underdose have a Target Gy less than 60 Gy, which will only punish the dose points below {target_gy} Gy. Please adjust it or give a rationale for the lower Target Gy.")

                if vol != 'NA' and '%' in vol:
                    raise ValueError(f"The % Volume parameter {vol} should not have a % symbol.")
                
                if obj_type in ['max_dose', 'mean_dose', 'dose_volume_V'] and 'NA' not in weight:
                    raise ValueError(f"The {obj_type} is a optimzation constraint and should not have a Weight parameter.")

                if obj_type in ['quadratic-overdose', 'quadratic-underdose', 'linear-overdose', 'quadratic', 'linear', 'smoothness-quadratic'] and 'NA' in weight:
                    raise ValueError(f"The {obj_type} is a optimization objective and should have a Weight parameter.")

        except Exception as e:
            error = True
            print(colored(f"An error occurred: {str(e)}", 'red'))
            if 'float' in str(e):
                critique = f'The OptPara markdown table is incomplete. Please generate OptPara Iter-{cur_iter} again.'
            else:
                critique = f'Errors or Warnings: {str(e)}. Consider generating OptPara Iter-{cur_iter} again.'
            # import pdb; pdb.set_trace()
            # print(f"debug .....")
            # re-generate the OptPara

            # sleep 5 minutes to avoid openai api rate limit
            # print(f"Sleeping for 1 minutes...")
            # time.sleep(1*60)

            msgs_to_send = messages + [ {'role': 'assistant', 'content': response}, {'role': 'user', 'content': critique} ] 
            response = self._generate_oai_reply_from_client(client, self._oai_system_message+msgs_to_send, self.client_cache)
            response = response['content'] if isinstance(response, dict) else response
            if 'Iter-' in response:
                response = re.sub(r"Iter-\d", "", response)  # remove the iteration number from the response

        compare = ''
        if False and not is_missing and len(invalid_ROIs)==0 and len(invalid_types)==0 and cur_iter>1 :
            # 2 compare OptPara versions
            prev_optPara = next((msg for msg in reversed(messages) if msg.get('role') == 'assistant'), None)
            assert prev_optPara is not None, "Previous OptPara not found in the chat history."
            prev_optPara = prev_optPara['content']
            msgs_to_send = [{'role': 'user', 'content': f'{prev_optPara}\n\nBelow is OptPara Iter-{cur_iter}:\n{response}'}]
            compare = self.agent_compareDosimetrist.generate_reply(msgs_to_send)['content']
            compare = f'\n\nBelow is the changes between OptPara Iter-{cur_iter-1} and Iter-{cur_iter}:\n{compare}'
            print(colored(compare, 'blue'))

        if False and cur_iter > 1:
            # 3 critique
            # prepare the msg history for critique. Only keep last two msgs: physicist's evaluation and human supervisor's feedback
            msgs = copy.deepcopy(messages)[-2:]
            assert msgs[-1]['name'] == 'human_supervisor' and msgs[-1]['role'] == 'user', "Human Supervisor's feedback not found in the chat history."
            assert msgs[-2]['name'] == 'physicist' and msgs[-2]['role'] == 'user', "Physicist's evaluation not found in the chat history."
            # let gemini know the roles of physicist and huamn supervisor
            msgs[-2]['content'] = f"\nBelow is Physicist Evaluation for OptPara Iter-{cur_iter}:\n" + msgs[-2]['content']
            msgs[-1]['content'] = f"\nBelow is Huamn Supervisor feedback for OptPara Iter-{cur_iter}:\n" + msgs[-1]['content']

            msgs_to_send = msgs + [{'role': 'user', 'content': f"Please review OptPara Iter-{cur_iter}:**\n{response}\n{compare}"}]
            #critique.append(self.agent_ctriticalDosimetrist.generate_reply(msgs_to_send)['content'])
            #print('\n\n', colored(critique[-1], 'blue'))
            
            critique = self.agent_ctriticalDosimetrist.generate_reply(msgs_to_send)['content']
            print('\n\n', colored(critique, 'blue'))
            
        if not error:
            break
    return response


def generate_oai_reply_with_process_physicist(
    self: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    """Generate a reply using autogen.oai."""
    client = self.client if config is None else config
    if client is None:
        return False, None
    if messages is None:
        messages = self._oai_messages[sender]  # sender always is the group manager

    assert self._name == 'physicist'

    # physicist_compare_with_protocol(self, messages)
    if not sender.is_skip_physicist:
        response = self._generate_oai_reply_from_client(client, self._oai_system_message + messages, self.client_cache)
        response = response['content'] if isinstance(response, dict) else response
        response = physicist_remove_repeat_phrase(response)
        response = physicist_post_process(self, messages, sender, response, client)

        # sleep at the end of the iteration
        if sender.optim_time < sender.iter_sleep:  # sleep 5 minutes to avoid openai api rate limit
            print(f"Sleeping for {sender.iter_sleep - sender.optim_time} seconds...")
            time.sleep(sender.iter_sleep - sender.optim_time)

    else:
        response = 'No comments.'

    physicist_save_trajectory(self, sender, messages)
    simplify_messages(self, sender, messages)
    response += f"\n\nNext, the dosimetrist should propose OptPara Iter-{sender.iter_count+1}."
    sender.iter_count += 1

    return True, response

def physicist_save_trajectory(self, sender, messages):
    '''return the optimization trajectories'''
    cur_iter = sender.iter_count
    if cur_iter > 0:
        dosimetrist_msg = next((msg for msg in reversed(messages) if msg.get('name') == 'dosimetrist'), None)
        tps_msg = next((msg for msg in reversed(messages) if msg.get('name') == 'TPS_proxy'), None)
        assert tps_msg is not None and dosimetrist_msg is not None, "TPS or dosimetrist's message not found in the chat history."

        traj_folder = f'{sender.traj_folder}/{sender.pid}'
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)

        # save the full optimization trajectories
        OptPara = extract_OptPara(cur_iter, dosimetrist_msg['content'])
        doseOut = tps_msg['content']
        new_traj = f"OptPara Iter-{cur_iter}:\n{OptPara}\n\nDosimetric OutComes:\n{doseOut}"
        sender.traj_full.append(new_traj)
        sender.traj_everyN.append(new_traj)
        with open(f'{traj_folder}/optim_trajectores_full.md', 'w') as f:
            f.write('\n\n'.join(sender.traj_full))

        # summarize the optimization trajectories
        if cur_iter%sender.everyN==0:  # every 10 iterations
            prefix = f'Now, analyze the trajectory from Iter-{cur_iter-sender.everyN+1} to Iter-{cur_iter}:\n'
            content = prefix + '\n\n'.join(sender.traj_everyN)  # current N traj
            sender.traj = self.agent_trajReviewer.generate_reply(messages=[{'role': 'user', 'content':content}])
            sender.traj = sender.traj['content'] if isinstance(sender.traj, dict) else sender.traj
            # traj_verf = self.agent_TrajVef.generate_reply(messages=[{'role': 'user', 'content':sender.traj}])
            # traj_verf = traj_verf['content'] if isinstance(traj_verf, dict) else traj_verf 
            # sender.traj += '\n\n' + traj_verf

            sender.traj_everyN = []  # reset
            with open(f'{traj_folder}/optim_trajectores.md', 'w') as f:
                f.write(sender.traj)
            print(colored(sender.traj, 'yellow'))

def physicist_compare_with_protocol(self, messages):
    dosimetric_outcomes = messages[-1]['content']
    comp_protocol = self.agent_comparePhysicist.generate_reply([messages[-1]])['content'] 
    comp_protocol = comp_protocol['content'] if isinstance(comp_protocol, dict) else comp_protocol
    print(colored(comp_protocol, 'blue'))
    # modify messages inplace to include comparision
    # messages[-1]['content'] = dosimetric_outcomes + '\n\n Protocol Adherence:\n' + comp_protocol

def physicist_post_process(
    self: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    response: str = None,
    client: Optional[OpenAIWrapper] = None,
) -> Union[str, Dict, None]:
    '''physicist self-reflection and correction'''

    critique = 'no errors'
    for i in range(3): 
        # print(colored(f'phycisist response-{i}', 'red'))
        # print(colored(response, 'blue'))

        # check the exsiting of Optimization Priorities
        is_missing = False if 'Optimization Priorities' in response else True
        if is_missing:
            critique = f'ERRORs: "### Optimization Priorities" is missing in your response! Please add this section to your response.'
            print(colored(critique, 'red'))
        
            msgs_to_send = messages + [ {'role': 'assistant', 'content': response}, {'role': 'user', 'content': critique} ] 
            response = self._generate_oai_reply_from_client(client, self._oai_system_message+msgs_to_send, self.client_cache)
        else:
            break

    return response

def physicist_remove_repeat_phrase(response):
    '''remove repeat phrase in response'''
    # split the response into lines
    lines = response.split('\n')
    lines = [l for l in lines if 'Next, the dosimetrist should propose OptPara' not in l]
    new_response = '\n'.join(lines)
    return new_response

def _simplify_physicist_msgs(self, messages, start_idx):
    """Compress the message history to keep only the key takeaways for all agents."""
    for msg in messages[start_idx:]:
        # Note: a agent's own msg in messages has role as assistant and name as None, so we get the agent owes name from self._name 
        speaker = msg.get('name', self._name)

        if 'physicist' in speaker or 'oncologist' in speaker:
            # keep only the content between 'begin_phrase' and "Now, the dosimetrist should propose OptPara"
            if 'Key Takeaways' in msg['content']:
                begin_phrase = 'Key Takeaways' 
            elif 'Optimization Priorities' in msg['content']:
                begin_phrase = 'Optimization Priorities'
            else:
                continue

            begin_idx = msg['content'].find(begin_phrase)
            end_idx = msg['content'].find('Now, the dosimetrist should propose OptPara')
            msg['content'] = msg['content'][begin_idx:end_idx]


def generate_TPS_reply(
    self: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    """Generate a reply using autogen.oai."""
    if messages is None:
        messages = self._oai_messages[sender]  # sender always is the group manager

    # try cache
    if not sender.traj_file is None:
        dose_out_md = parse_traj(sender.traj_file, sender.iter_count, 'tps')
        if not dose_out_md is None:
            sender.is_skip_physicist = True
            return True, dose_out_md

    sender.is_skip_physicist = False

    # get the optimization parameters from the dosimetrist
    assert messages[-1]['name'] == 'dosimetrist' and messages[-1]['role'] == 'user' 
    response_w_optPara = messages[-1]['content']  # dosimetrist's response with OptPara
    optPara_md = extract_OptPara(sender.iter_count, response_w_optPara)
    op_json = self.j2m.markdown_to_json(optPara_md)

    solution, dose_out_md, dose_out_df, elapsed_time = self.imrt.do_optim(op_json)
    sender.dose_out_df = dose_out_df
    sender.optim_time = elapsed_time

    # save solution to pickle file
    traj_folder = f'{sender.traj_folder}/{sender.pid}'
    if not os.path.exists(traj_folder):
        os.makedirs(traj_folder)
    with open(f'{traj_folder}/solution_x_{sender.iter_count}.pkl', 'wb') as f:
        pickle.dump(solution['optimal_intensity'], f)

    # handle the optimization complexity
    response = f"Given OptPara Iter-{sender.iter_count}, the dosimetric outcomes produced by TPS are:\n {dose_out_md}"
    if elapsed_time > sender.optim_time_limit-1:  # 20 minutes
        d95 = dose_out_df.loc[dose_out_df['Struct'] == 'PTV', 'Achieved Value'].values[0]
        if d95 < 50:
            response += "\nTPS Error: Optimization cannot solve the optimization problem within the time limit, resulting in an erroneous solution. Immediate action is required. You should simplify OptPara, especially by reducing the dose_volume_V constraints, as they can significantly increase the optimization complexity."
        else:
            response += "\nTPS Warning: Optimization cannot solve the optimization problem within the time limit, resulting in a suboptimal solution. This may be due to the complexity of dose_volume_V constraints. You may consider simplifying OptPara to reduce the optimization complexity, but you can keep the current OptPara with caution or awareness of the complexity."

    return True, response

def generate_default_humanSupervisor_reply(
    self: ConversableAgent,
    messages: Optional[List[Dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> Tuple[bool, Union[str, Dict, None]]:
    client = self.client if config is None else config
    if messages is None:
        messages = self._oai_messages[sender]  # sender always is the group manager
    assert self._name == 'human_supervisor', 'Only human_supervisor agent can use this function'

    extracted_response = 'No comments.'

    return (False, None) if extracted_response is None else (True, extracted_response)

def test_parse_traj():
    traj_file_path = './debug/Lung_Patient_42/optim_trajectories_full.md'
    iter_idx = 3 
    role = 'dosimetric'
    print(parse_traj(traj_file_path, iter_idx, role))
    role = 'TPS'
    print(parse_traj(traj_file_path, iter_idx, role))

def test_dosimetrist_extract_OptPara():
    response = textwrap.dedent("""\
### Updated OptPara Iter-21
```
| ROI Name      | Objective Type       | Target Gy | % Volume | Weight   |
|:--------------|:---------------------|:----------|:---------|:---------|
| PTV           | quadratic-overdose   | 60        | NA       | 15000    |
| PTV           | quadratic-underdose  | 60        | NA       | 620000   |
| PTV           | quadratic-underdose  | 60.5      | NA       | 460000   |
| PTV           | dose_volume_V        | 60        | 99       | NA       |
| PTV           | max_dose             | 68        | NA       | NA       |
| GTV           | max_dose             | 68        | NA       | NA       |
| CORD          | linear-overdose      | 40        | NA       | 1600     |
| CORD          | quadratic            | NA        | NA       | 15       |
| CORD          | max_dose             | 47        | NA       | NA       |
| ESOPHAGUS     | quadratic            | NA        | NA       | 20       |
| ESOPHAGUS     | max_dose             | 60        | NA       | NA       |
| ESOPHAGUS     | mean_dose            | 34        | NA       | NA       |
| HEART         | quadratic            | NA        | NA       | 20       |
| HEART         | max_dose             | 60        | NA       | NA       |
| HEART         | mean_dose            | 25        | NA       | NA       |
| HEART         | dose_volume_V        | 30        | 40       | NA       |
| LUNGS_NOT_GTV | quadratic            | NA        | NA       | 20       |
| LUNGS_NOT_GTV | max_dose             | 63        | NA       | NA       |
| LUNGS_NOT_GTV | mean_dose            | 16        | NA       | NA       |
| LUNGS_NOT_GTV | dose_volume_V        | 20        | 30       | NA       |
| LUNG_L        | quadratic            | NA        | NA       | 10       |
| LUNG_L        | max_dose             | 63        | NA       | NA       |
| LUNG_R        | quadratic            | NA        | NA       | 10       |
| LUNG_R        | max_dose             | 63        | NA       | NA       |
| RIND_0        | quadratic            | NA        | NA       | 12       |
| RIND_0        | max_dose             | 66        | NA       | NA       |
| RIND_1        | quadratic            | NA        | NA       | 12       |
| RIND_1        | max_dose             | 63        | NA       | NA       |
| RIND_2        | quadratic            | NA        | NA       | 7        |
| RIND_2        | max_dose             | 54        | NA       | NA       |
| RIND_3        | quadratic            | NA        | NA       | 7        |
| RIND_3        | max_dose             | 51        | NA       | NA       |
| RIND_4        | quadratic            | NA        | NA       | 7        |
| RIND_4        | max_dose             | 45        | NA       | NA       |
| SKIN          | max_dose             | 60        | NA       | NA       |
| NA            | smoothness-quadratic | NA        | NA       | 1800     |
```
""")
    dosimetrist_post_process(None, 48, [], response, None)    

if __name__ == '__main__':
    test_dosimetrist_extract_OptPara()
