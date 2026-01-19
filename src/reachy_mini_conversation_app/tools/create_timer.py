import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CreateTimer(Tool):
    """Create a countdown timer that notifies when time is up."""

    name = "create_timer"
    description = """Create a countdown timer. When the timer expires, Reachy will notify you.
    Examples:
    - "Set a timer for 5 minutes"
    - "Timer for 30 seconds"
    - "Set a 2 minute timer"
    """
    parameters_schema = {
        "type": "object",
        "properties": {
            "duration_seconds": {
                "type": "number",
                "description": "Duration in seconds (e.g., 300 for 5 minutes, 60 for 1 minute)",
            },
            "label": {
                "type": "string",
                "description": "Optional label for the timer (e.g., 'pizza', 'workout')",
            },
        },
        "required": ["duration_seconds"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Create a timer."""
        duration = kwargs.get("duration_seconds")
        label = kwargs.get("label", "")
        
        if not duration or duration <= 0:
            return {"error": "Duration must be a positive number"}
        
        duration = float(duration)
        
        # Get task manager from deps (we'll need to add this)
        if not hasattr(deps, 'task_manager') or deps.task_manager is None:
            return {"error": "Task manager not available"}
        
        task_id = deps.task_manager.create_timer(int(duration))
        
        # Format duration nicely
        if duration < 60:
            duration_str = f"{int(duration)} seconds"
        elif duration < 3600:
            minutes = int(duration // 60)
            duration_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            if minutes > 0:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
            else:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''}"
        
        logger.info(f"Tool call: create_timer duration={duration_str} label={label}")
        
        return {
            "task_id": task_id,
            "duration": duration_str,
            "label": label,
            "status": "Timer started"
        }