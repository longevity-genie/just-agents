from just_agents.interfaces.agent import *

class ITypedAgent(
    IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
    Generic[
        AbstractQueryInputType,
        AbstractQueryResponseType,
        AbstractStreamingChunkType,
        AbstractAgentInputType,
        AbstractAgentOutputType
    ]
):
    """
    A typed agent is an agent that can query and stream typed inputs and outputs.
    It can recieve typed inputs from which it will extract query for the LLM and then return typed outputs.
    """
    

    @abstractmethod
    def query_from_input(self, input: AbstractAgentInputType) -> AbstractQueryInputType:
        """
        This method is used to convert the input to the query input type for self.query(message) call
        """
        raise NotImplementedError("You need to implement query_from_input() abstract method first!")
    
    @abstractmethod
    def output_from_response(self, response: AbstractQueryResponseType) -> AbstractAgentOutputType:
        """
        This method is used to convert the query response to the output type.
        
        Args:
            response: The raw response from the query
            
        Returns:
            The response converted to the agent's output type
        """
        # Get the concrete output type from the class's type parameters
        output_type = self.__class__.__orig_bases__[0].__args__[4]  # Gets AbstractAgentOutputType
        return self.query_structural(response, parser=output_type)
        
    
    def query_typed(self, input: AbstractAgentInputType) -> AbstractAgentOutputType:
        """
        This method is used to query the agent with a typed input and get a typed output.
        """
        query = self.query_from_input(input)
        response = self.query(query)
        output = self.output_from_response(response)
        return output