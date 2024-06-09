from dataclasses import dataclass, field, replace
from dataclasses import asdict
from typing import Any, Dict, Optional


from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class LLMOptions:
    """
    Class for additional LLM options
    """
    model: str
    api_base: Optional[str] = None
    temperature: float = 0.0
    extras: Dict[str, Any] = field(default_factory=lambda: {})
    tools: list = field(default_factory=lambda : [])
    tool_choice: Optional[str] = field(default_factory=lambda: None) #"auto"

    def to_dict(self):
        data = asdict(self)
        extras = data.pop('extras', {})
        return {**data, **extras}

    def __getitem__(self, key):
        if key in self.extras:
            return self.extras[key]
        return getattr(self, key)

    def copy(self, **changes):
        return replace(self, **changes)

LLAMA3: LLMOptions = LLMOptions("groq/llama3-70b-8192", "https://api.groq.com/openai/v1")
OPEN_ROUTER_Qwen_2_72B_Instruct: LLMOptions = LLMOptions("openrouter/qwen/qwen-2-72b-instruct", "https://openrouter.ai/api/v1")
TOGETHER_Qwen_2_72B_Instruct: LLMOptions = LLMOptions("together_ai/Qwen/Qwen2-72B-Instruct")
FIREWORKS_Qwen_2_72B_Instruct: LLMOptions = LLMOptions("fireworks_ai/wen/Qwen2-72B-Instruct")