from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool

from src.config import config
from src.llm import get_synthesis_llm
from src.logger import get_logger, timer
from src.state import AgentState


class BaseWorker:
    """
    Template for a worker node.

    Subclass this, set `name`, `system_prompt`, and optionally `tools`,
    then use the instance's `run()` method as the graph node function body.
    """

    name: str = "worker"
    system_prompt: str = "You are a helpful assistant."
    tools: list[BaseTool] = []
    use_small_model: bool = False  # set True to route this worker to the local model

    def _build_llm(self) -> BaseChatModel:
        if self.use_small_model:
            llm = get_synthesis_llm()
        else:
            llm = ChatGroq(model=config.MODEL_NAME, api_key=config.GROQ_API_KEY)
        return llm.bind_tools(self.tools) if self.tools else llm

    def run(self, state: AgentState) -> dict:
        """Execute this worker's task and return updated state fields."""
        log = get_logger(f"src.workers.{self.name}")
        log.info("worker entered", extra={"worker": self.name})

        llm = self._build_llm()
        instructions = state.get("instructions", "Complete your assigned task.")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=instructions),
        ]
        with timer(log, "llm_call", worker=self.name):
            response = llm.invoke(messages)

        log.info("worker completed", extra={"worker": self.name, "output_len": len(response.content)})
        output = f"[{self.name}]\n{response.content}"
        return {"worker_outputs": [output]}

    def as_node(self):
        """Return a plain function suitable for graph.add_node()."""
        def node(state: AgentState) -> dict:
            return self.run(state)
        node.__name__ = self.name
        return node
