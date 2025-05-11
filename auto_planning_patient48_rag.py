from pathlib import Path
import sys, os
from agent_team import AgentTeam
from database_Lung import LungCancerDB
from rag import RAGLung

if __name__ == '__main__':
    # setup DB
    db = LungCancerDB()
    print("all patient names in DB:")
    print(db.get_all_patient_names())
    print("\n\n")

    # setup RAG based on DB
    rag = RAGLung(db, top_num=3)

    # setup agents 
    log_dir = Path('/mnt/disk16T/datasets/PortPy_datasets/traj_files_rag')
    traj_file = None
    # traj_file = log_dir/Path('Lung_Patient_48/optim_trajectores_full_cp.md')
    team = AgentTeam(db, rag, traj_folder=log_dir, traj_file=traj_file, optim_time_limit=5*60, iter_sleep=1*60,
                      traj_llm_tag='gm-2.5-flash', sug_llm_tag='gm-2.5-flash', dos_llm_tag='gm-2.5-flash', phy_llm_tag='gm-2.5-flash')

    # setup planning task
    pid = 'Lung_Patient_48'
    init_msg = team.get_initmsg(patient_name=pid)

    # run auto planning
    team.run_auto_planning(init_msg) 