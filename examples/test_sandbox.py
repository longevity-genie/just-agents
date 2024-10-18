from just_agents_sandbox.micromamba_session import MicromambaSession

# Create a new sandbox session
def test_sandbox():
    # Or default image
    with  MicromambaSession(image="quay.io/longevity-genie/biosandbox:latest", lang="python", keep_template=True, verbose=True) as session:
        session.execute_command("python --version")
        result = session.run("print('Hello, World!')")
        

test_sandbox()