from typing import Optional
import docker
from llm_sandbox.session import SandboxDockerSession
from llm_sandbox.docker import ConsoleOutput
from llm_sandbox.const import SupportedLanguage
from docker.types import Mount

class MicromambaSession(SandboxDockerSession):
    """
    Docker session for running micromamba in docker container
    """
    def __init__(self, client: Optional[docker.DockerClient] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False, 
        environment: str = "base",
        mounts: Optional[list[Mount]] = None,
        ):
        super().__init__(client=client,
                         image=image,
                         dockerfile=dockerfile,
                         lang=lang,
                         keep_template=keep_template,
                         verbose=verbose, 
                         mounts=mounts)
        self.environment = environment


    def execute_command(
        self, command: Optional[str], workdir: Optional[str] = None
    ) -> ConsoleOutput:
        if not command:
            raise ValueError("Command cannot be empty")

        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before executing commands."
            )
        command = f"micromamba run -n {self.environment} {command}"

        if self.verbose:
            print(f"Executing command: {command}")

        if workdir:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True, workdir=workdir
            )
        else:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True
            )

        output = ""
        if self.verbose:
            print("Output:", end=" ")

        for chunk in exec_log:
            chunk_str = chunk.decode("utf-8")
            output += chunk_str
            if self.verbose:
                print(chunk_str, end="")

        return ConsoleOutput(output)
