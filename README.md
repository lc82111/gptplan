# gptplan
This project, `gptplan`, is designed for automated radiotherapy treatment planning using advanced AI techniques. It integrates multi-agent systems, retrieval augmented generation, and specialized radiotherapy libraries to streamline and potentially optimize the planning process.

## Core Functionality
The system uses AI agents, built with the `pyautogen` framework, to automate various aspects of treatment planning. Key functionalities include:
*   **Patient Data Management:** Interacts with patient databases (e.g., `database_Lung.py` for lung cancer cases) to retrieve and manage necessary data.
*   **AI Agent Collaboration:** Employs a team of AI agents (`agent_team.py`) that collaborate on planning tasks, guided by predefined prompts (`prompts.py`) and LLM configurations (`llm_config.py`).
*   **Retrieval Augmented Generation (RAG):** Utilizes RAG techniques (`rag.py`) to provide agents with relevant information from a knowledge base, enhancing the quality and context-awareness of their decisions.
*   **Radiotherapy Planning with PortPy:** Leverages the `portpy` library (`use_portpy.py`) for core radiotherapy calculations, plan optimization, and simulation, using specific configuration files found in `portpy_config_files/`.

## Dependencies
The project is built in Python and relies on several external libraries.

*   **Python 3.x**
*   **Key Libraries:**
    *   `pyautogen`: For building and managing AI agent teams. (Note: `pyautogen` has undergone significant changes. The version specified in `requirements.txt` (`0.2.33`) is recommended to ensure compatibility with this project.)
    *   `portpy`: For radiotherapy planning, optimization, and dose calculation.
    *   `google-generativeai`: For interacting with Google's Generative AI models.
    *   `pandas`, `numpy`, `scipy`: For data manipulation and scientific computing.
    *   `cvxpy`: For optimization.
    *   `matplotlib`, `seaborn`: For plotting and visualization.
    *   `scikit-image`: For image processing.
    *   And others as listed in `requirements.txt`.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    # git clone <repository_url>
    # cd gptplan
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    pip install -r requirements.txt
    ```

3.  **LLM Configuration:**
    The project uses Large Language Models (LLMs) via `pyautogen`. You need to configure your LLM API keys.
    *   Examine `llm_config.py`. This file likely sets up the configuration for the LLM provider (e.g., Google Generative AI, given the `google-generativeai` dependency).
    *   You may need to create or update `llm_config.json` or set environment variables with your API keys as required by `llm_config.py` and `pyautogen`. Refer to the `pyautogen` documentation for details on configuring LLM providers.

## How to Run

The primary way to run a full auto-planning workflow is likely through the main execution scripts. Individual modules also contain test routines.

1.  **Main Auto-Planning Workflow:**
    The script `auto_planning_patient48_rag.py` appears to be an example of an end-to-end automated planning process.
    ```bash
    python auto_planning_patient48_rag.py
    ```
    Ensure your LLM configuration is correctly set up before running. This script will likely demonstrate the collaboration of agents, RAG, and PortPy for a specific patient case.

2.  **Running Individual Modules / Unit Tests:**
    Several modules have `if __name__ == "__main__":` blocks, often calling `unit_test()` functions. These can be run directly to test or demonstrate the functionality of that specific module:
    *   `python agent_team.py`
    *   `python database_Lung.py`
    *   `python rag.py`
    *   `python use_portpy.py`

## Project Structure Overview

*   `agent_team.py`: Defines and manages the team of AI agents.
*   `auto_planning_patient48_rag.py`: Example script for running the full auto-planning workflow with RAG.
*   `database.py` / `database_Lung.py`: Handles database interactions and patient data.
*   `llm_config.py`: Configures the Large Language Models for `pyautogen`.
*   `llm_config.json`: (If used) Stores LLM API keys or other related configurations.
*   `prompts.py`: Contains the prompts used to guide the AI agents.
*   `rag.py`: Implements the Retrieval Augmented Generation logic.
*   `use_portpy.py`: Demonstrates and utilizes functionalities of the `portpy` library.
*   `utils.py`: Contains utility functions used across the project.
*   `portpy_config_files/`: Directory containing configuration files for `portpy`, such as clinical criteria and optimization parameters.
*   `requirements.txt`: Lists all Python package dependencies.
*   `README.md`: This file.

This README provides a starting point. You may need to add more specific details about the data formats, particular patient cases, or advanced configuration options as the project evolves.
