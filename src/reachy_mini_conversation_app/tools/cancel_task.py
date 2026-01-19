import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CancelTask(Tool):
    """Cancel a timer or reminder."""

    name = "cancel_task"
    description = "Cancel an active timer or reminder by its ID"
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel (e.g., 'timer_1', 'reminder_1')",
            },
        },
        "required": ["task_id"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a task."""
        task_id = kwargs.get("task_id", "").strip()
        
        if not task_id:
            return {"error": "Task ID is required"}
        
        if not hasattr(deps, 'task_manager') or deps.task_manager is None:
            return {"error": "Task manager not available"}
        
        success = deps.task_manager.cancel_task(task_id)
        
        logger.info(f"Tool call: cancel_task task_id={task_id} success={success}")
        
        if success:
            return {"status": f"Cancelled task {task_id}"}
        else:
            return {"error": f"Task {task_id} not found or already completed"}