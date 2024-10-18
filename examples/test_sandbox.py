from llm_sandbox import SandboxSession

# Create a new sandbox session
def test_sandbox():
    with SandboxSession(image="mambaorg/micromamba:latest", keep_template=True) as session:
        result = session.run("echo 'Hello world'")
        print(result)


test_sandbox()