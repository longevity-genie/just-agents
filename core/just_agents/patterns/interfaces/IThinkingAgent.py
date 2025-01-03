from just_agents.interfaces.agent import *
from typing import ClassVar, Any

# New TypeVar for Thought
THOUGHT_TYPE = TypeVar('THOUGHT_TYPE', bound='IThought')

class IThought(BaseModel):
    content: Any #

    @abstractmethod
    def is_final(self) -> bool:
        raise NotImplementedError("You need to implement is_final() abstract method first!")

class ErrorThought(IThought):
    def is_final(self) -> bool:
        # Define logic to determine if this thought is final
        return True  # For error thoughts, consider them final

class IThinkingAgent(
    IAgent[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType],
    Generic[AbstractQueryInputType, AbstractQueryResponseType, AbstractStreamingChunkType, THOUGHT_TYPE]
):
    MAX_STEPS: ClassVar[int] = 8

    @abstractmethod
    def thought_query(self, response: AbstractQueryInputType, **kwargs) -> THOUGHT_TYPE:
        raise NotImplementedError("You need to implement thought_query abstract method first!")

    def think(self, 
              query: AbstractQueryInputType, 
              max_iter: Optional[int] = None,
              chain: Optional[list[THOUGHT_TYPE]] = None,
              **kwargs  ) -> tuple[Optional[THOUGHT_TYPE], Optional[list[THOUGHT_TYPE]]]:
        """
        This method will continue to query the agent until the final thought is not None or the max_iter is reached.
        Returns a tuple of (final_thought, thought_chain)
        """
        if not max_iter:
            max_iter = IThinkingAgent.MAX_STEPS
        if max_iter < 1:
            return (None,None)
        current_chain = list(chain) if chain else [] #shallow copy chain rather than modifying mutable instance
        for step in range(max_iter):
            try:
                thought = self.thought_query(query, **kwargs) # queries itself with thought as expected output
            except Exception as e:
                return (
                    ErrorThought(content=f"Error during thought_query at step {step + 1}: {e}"),
                    current_chain
                )

            current_chain.append(thought) # updates chain with the new thought
            if thought.is_final() or step == max_iter-1:
                return (
                    thought,
                    current_chain
                ) #returns the final thought and the chain that preceded it

