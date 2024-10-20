from just_agents_sandbox.micromamba_session import MicromambaSession
from llm_sandbox.docker import ConsoleOutput

def run_bash_command(command: str):
    """
    command: str # command to run in bash, for example install software inside micromamba environment
    """
    with  MicromambaSession(image="quay.io/longevity-genie/biosandbox:latest", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.execute_command(command=command)
        return result
    

def run_python_code(code: str):
    """
    code: str # python code to run in micromamba environment
    """
    with  MicromambaSession(image="quay.io/longevity-genie/biosandbox:latest", lang="python", keep_template=True, verbose=True) as session:
        result: ConsoleOutput = session.run(code)
        return result