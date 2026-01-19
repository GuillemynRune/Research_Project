import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ListTasks(Tool):
    """List all active timers and reminders."""

    name = "list_tasks"
    description = "Show all active timers and reminders"
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """List all active tasks."""
        if not hasattr(deps, 'task_manager') or deps.task_manager is None:
            return {"error": "Task manager not available"}
        
        active_tasks = deps.task_manager.get_active_tasks()
        
        logger.info(f"Tool call: list_tasks (found {len(active_tasks)} active)")
        
        tasks_info = []
        for task in active_tasks:
            status = deps.task_manager.get_task_status(task.task_id)
            if status:
                tasks_info.append(status)
        
        return {
            "active_count": len(active_tasks),
            "tasks": tasks_info
        }