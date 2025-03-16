from just_agents.just_bus import JustLogBus
from just_agents.base_agent import BaseAgent
from just_agents.just_locator import JustAgentsLocator
from typing import Optional

def call_expert_agent(agent_name: str, user_query: str, agent_codename: Optional[str] = None, call_the_first_instance: bool = True) -> str:
    """
    Call the expert agent with the given name.

    Args:
        agent_name: The name of the agent to call
        user_query: The query to pass to the agent
        agent_codename: The unique codename of the agent instance to select if multiple instances of the same agent are found
        call_the_first_instance: Whether to call the first instance of the agent, true by default
        
    Returns:
        The response from the agent or the error message
    """
    locator = JustAgentsLocator()
    log_bus = JustLogBus()

    agents = locator.get_agents_by_shortname(agent_name, bounding_class=BaseAgent)
    if not agents:
        return f"Agent with shortname {agent_name} not found"
    
    agent = agents[0]

    if len(agents) > 1 and not call_the_first_instance:
        codenames = locator.get_codenames_by_shortname(agent_name, bounding_class=BaseAgent)
        log_bus.debug(f"Multiple agents with shortname {agent_name} found",
                        source="call_expert_agent",
                        action="agent_locator.get_agents_by_shortname",
                        codenames=codenames,
                        agent_name=agent_name,
                        call_the_first_instance=call_the_first_instance,
                        agent_codename=agent_codename)

        if agent_codename is None:
            return f"Multiple agents with shortname {agent_name} found, codenames: {str(codenames)}"
        else:
            if agent_codename not in codenames:
                return f"Agent with codename {agent_codename} not found, existing codenames: {str(codenames)}"
            else:
                agent = locator.get_agent_by_codename(agent_codename)
    
    return f"Agent response: {agent.query(user_query)}"

