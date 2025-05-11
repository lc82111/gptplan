import pprint, os, json, sqlite3, re, textwrap
import autogen
from autogen.agentchat import ConversableAgent, UserProxyAgent, Agent, AssistantAgent, GroupChat, GroupChatManager
from autogen.oai import config_list_from_json, filter_config
from autogen import runtime_logging
import numpy as np
import pandas as pd
from database import CervicalCancerDB
from database_Lung import LungCancerDB
from rag import RAGLung, RAGLung
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
from utils import generate_default_humanSupervisor_reply, generate_oai_reply_with_process_dosimetrist, generate_TPS_reply, generate_oai_reply_with_process_physicist
from prompts import get_iniTaskMsg_portpy_0727, get_iniTaskMsg_portpy_0806, get_prompt, get_iniTaskMsg_portpy, get_iniTaskMsg_portpy_0710
from use_portpy import IMRT, JSON2Markdown 

if False:
    os.environ["http_proxy"] = "http://127.0.0.1:8019"
    os.environ["https_proxy"] = "http://127.0.0.1:8019"
    os.environ["socks_proxy"] = "socks5://127.0.0.1:8018"
else:
    os.environ["http_proxy"] = "http://127.0.0.1:20171"
    os.environ["https_proxy"] = "http://127.0.0.1:20171"
    os.environ["socks_proxy"] = "socks5://127.0.0.1:20170"

# config_list = config_list_from_json(env_or_file="llm_config.json")

from llm_config import config_list
class AgentTeam:
    def __init__(self, db: LungCancerDB, rag: RAGLung, traj_folder, traj_file=None, optim_time_limit=1200, iter_sleep=300,
                  traj_llm_tag='gm-pro', sug_llm_tag='llama405B', dos_llm_tag='local-gpt-4o', phy_llm_tag='local-gpt-4o',
                  criteria_json_fn='./portpy_config_files/clinical_criteria/Default/Lung_2Gy_30Fx_wqx.json'):
        self.db = db
        self.rag = rag
        self.pid = None
        self.traj_folder = traj_folder
        self.traj_file = traj_file
        self.optim_time_limit = optim_time_limit
        self.criteria_json_fn = criteria_json_fn
        self.iter_sleep = iter_sleep
        self.traj_llm_tag = traj_llm_tag
        self.sug_llm_tag = sug_llm_tag
        self.dos_llm_tag = dos_llm_tag
        self.phy_llm_tag = phy_llm_tag
    
    def get_initmsg(self, patient_name):
        '''patient_name: patient who needs treatment plan.  '''
        self.pid = patient_name
        # init_msg = get_iniTaskMsg_portpy_0727(patient_name)
        query_anatomy_str = self.rag.get_patient_anatomy_str(patient_name)
        ref_plans_str = self.rag.get_ref_plans(patient_name)
        init_msg = get_iniTaskMsg_portpy_0806(patient_name, query_anatomy_str, ref_plans_str, None)
        return init_msg

    def set_initializer(self):
        # initializer agent
        return UserProxyAgent(
            name="task_initializer",
            llm_config=False,
            code_execution_config=False,
            description="Initiate the planning process by providing patient details, prescribed dose, dose objectives, constraints, and any relevant reference plans or optimization parameters to guide the team."
        )

    def set_tps(self):
        # tps agent
        agent = ConversableAgent(
            name="TPS_proxy",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="NEVER",
            description= "A TPS proxy simulates and optimizes the plan based on the set of optimization parameters."
        )
        agent.register_reply([Agent, None], generate_TPS_reply)  # set to 2 for huamn input first

        # additional attributes for optimization
        agent.imrt = IMRT(pid=self.pid, time_limit=self.optim_time_limit, criteria_json_fn=self.criteria_json_fn)
        agent.j2m = JSON2Markdown() 

        return agent

    def set_dosimetrist(self):
        # dosimetrist agent
        agent = AssistantAgent(
            name="dosimetrist",
            llm_config=filter_config(config_list, {"tags":[self.dos_llm_tag]})[0],
            system_message=get_prompt('sysmsg_dosimetris_portpy_rag_0807'),
            human_input_mode="NEVER",
            description="A senior dosimetrist proposes OptPara for TPS.",
        )
        agent.register_reply([Agent, None], generate_oai_reply_with_process_dosimetrist, position=4)  # set to 4 for human input first

        agent.agent_compareDosimetrist = ConversableAgent(
            name="Comparing Dosimetrist",
            system_message=get_prompt('sysmsg_CompareDosimetrist_portpy_0717'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_OptParaCompleteChecker = ConversableAgent(
            name="OptParaCompleteChecker",
            system_message=get_prompt('sysmsg_OptParaCompleteChecker_gemini_0721'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_ctriticalDosimetrist = ConversableAgent(
            name="Critical Dosimetrist",
            system_message=get_prompt('sysmsg_criticalDosimetrist_gemini_portpy_0710'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_SuggestDosimetrist = ConversableAgent(
            name="Suggest Dosimetrist",
            system_message=get_prompt('sysmsg_SuggestDosimetrist_gpt4_portpy_0719'),
            llm_config=filter_config(config_list, {"tags":[self.sug_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_SuggestDosimetristCheck = ConversableAgent(
            name="SuggestDosimetristCheck",
            system_message=get_prompt('sysmsg_SuggestDosimetristCheck_portpy_0719'),
            llm_config=filter_config(config_list, {"tags":["gm-pro"]})[0],
            human_input_mode="NEVER"
        )

        return agent

    def set_physicist(self):
        # physicist agent
        agent = AssistantAgent(
            name="physicist",
            system_message=get_prompt('sysmsg_physicist_portpy_0715'),
            llm_config=filter_config(config_list, {"tags":[self.phy_llm_tag]})[0],
            description="A senior medical physicist evaluates the plan from a technical perspective",
        )
        agent.register_reply([Agent, None], generate_oai_reply_with_process_physicist)

        agent.agent_comparePhysicist = ConversableAgent(
            name="Comparing Physicist",
            system_message=get_prompt('sysmsg_ComparePhysicist_portpy_0712'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_trajReviewer = ConversableAgent(
            name="Trajectory Dosimetrist",
            system_message=get_prompt('sysmsg_Trajectory_gemini_portpy_0731'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        agent.agent_TrajVef = ConversableAgent(
            name="trajectoryVefDosimetrist",
            system_message=get_prompt('sysmsg_TrajVefDosimetrist_portpy_0719'),
            llm_config=filter_config(config_list, {"tags":[self.traj_llm_tag]})[0],
            human_input_mode="NEVER"
        )

        return agent

    def set_humanSupervisor(self):
        # human supervisor agent
        agent = UserProxyAgent(
            name="human_supervisor",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="ALWAYS",
            description="A human supervisor provides guidance and feedback to the team members during the planning process."
        )
        agent.register_reply([Agent, None], generate_default_humanSupervisor_reply, position=2)  # set to 2 for check huamn input first
        # print(agent._reply_func_list)
        return agent

    def set_oncologist(self):
        agent = AssistantAgent(
            name="oncologist",
            system_message=get_prompt('sysmsg_oncologist_0526'),
            llm_config=filter_config(config_list, {"tags":["local-gpt-4o"]})[0],
            description="A senior radiation oncologist reviews the plan from a clinical perspective.",
        )
        agent.register_reply([Agent, None], generate_oai_reply_with_process_dosimetrist)
        return agent

    def set_team(self):
        # set all team members 
        initializer = self.set_initializer()
        dosimetrist = self.set_dosimetrist()
        tps_proxy = self.set_tps()
        physicist = self.set_physicist()
        human_supervisor = self.set_humanSupervisor()
        oncologist = self.set_oncologist()
        # agents = [initializer, dosimetrist, tps_proxy, physicist, oncologist]
        agents = [initializer, dosimetrist, tps_proxy, physicist, human_supervisor]       # agents = [initializer, dosimetrist, tps_proxy, physicist]

        # set allowed transitions
        allowed_transitions = {
            initializer: [dosimetrist],
            dosimetrist: [tps_proxy],
            tps_proxy: [physicist],
            physicist: [dosimetrist],
            # physicist: [oncologist],
            # oncologist: [dosimetrist],
        }

        allowed_transitions = {
            initializer: [dosimetrist],
            dosimetrist: [tps_proxy],
            tps_proxy: [physicist],
            physicist: [human_supervisor],
            human_supervisor: [dosimetrist],
        }
 
        groupchat = GroupChat(agents=agents,
                                    speaker_selection_method="round_robin",
                                    allowed_or_disallowed_speaker_transitions=allowed_transitions,
                                    speaker_transitions_type="allowed",
                                    messages=[], max_round=1000, send_introductions=True)
        groupchat.DEFAULT_INTRO_MSG = get_prompt('groupchat_intromsg_0710')
        manager = GroupChatManager(groupchat=groupchat, llm_config=False,
                                            system_message="You are the group chat manager overseeing the radiotherapy treatment planning process. Monitor the interactions between agents, provide guidance, and ensure the completion of the planning task within the specified rounds.")
        
        # set manager attributes to share with agents
        manager.pid = self.pid
        manager.iter_count = 1
        manager.traj = ''
        manager.traj_full = []
        manager.traj_everyN = []
        manager.everyN = 5 
        manager.traj_folder = self.traj_folder
        manager.traj_file = self.traj_file
        manager.optim_time_limit = self.optim_time_limit
        manager.iter_sleep = self.iter_sleep
        manager.is_skip_physicist = False  # if OptPara and Dose outcomes can be found in file, skip physicist

        return initializer, manager        

    def run_auto_planning(self, init_msg):
        initializer, manager = self.set_team()

        logging_session_id = runtime_logging.start(config={"dbname": ".logs_lung_new.db"})

        initializer.initiate_chat(manager, message=init_msg)

        runtime_logging.stop()

    @staticmethod
    def unit_test():
        # setup DB
        db = CervicalCancerDB()

        # setup RAG based on DB
        rag = RAGLung(db)

        # setup autoplanning team
        team = AgentTeam(db, rag)

        # setup planning task
        init_msg = team.get_initmsg(patient_name="D", top_n=2)

        # run auto planning
        team.run_auto_planning(init_msg) 


if __name__ == '__main__':
    AgentTeam.unit_test()