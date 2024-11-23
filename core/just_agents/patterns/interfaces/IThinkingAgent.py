from just_agents.core.interfaces.IAgent import *

# New TypeVar for Thought
THOUGHT_TYPE = TypeVar('THOUGHT_TYPE', bound='IThought')

class IThought(BaseModel):
    @abstractmethod
    def is_final(self) -> bool:
        raise NotImplementedError("You need to implement is_final() abstract method first!")


class IThinkingAgent(
    IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
    Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType, THOUGHT_TYPE]
):
    
    @abstractmethod
    def thought_query(self, response: AbstractQueryResponseType, **kwargs) -> THOUGHT_TYPE:
        raise NotImplementedError("You need to implement thought_from_response() abstract method first!")

    def think(self, 
              query: AbstractQueryInputType, 
              max_iter: int = 3, 
              chain: Optional[list[THOUGHT_TYPE]] = None,
              **kwargs  ) -> tuple[Optional[THOUGHT_TYPE], Optional[list[THOUGHT_TYPE]]]:
        """
        This method will continue to query the agent until the final thought is not None or the max_iter is reached.
        Returns a tuple of (final_thought, thought_chain)
        """
        current_chain = chain or []
        thought = self.thought_query(query, **kwargs) #queries itself with thought as expected output
        new_chain = [*current_chain, thought] #updates chain with the new thought
        if thought.is_final() or max_iter <= 0:
            return (self.thought_query(query, **kwargs), new_chain) #returns the final thought and the chain that preceded it
        else:
            return self.think(query, max_iter - 1, new_chain, **kwargs) #continues the thought process