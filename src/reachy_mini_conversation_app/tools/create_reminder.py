import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CreateReminder(Tool):
    """Create a reminder that notifies after a delay."""

    name = "create_reminder"
    description = """Create a reminder with a custom message. Reachy will remind you after the specified time.
    Examples:
    - "Remind me to call mom in 1 hour"
    - "Set a reminder to take medicine in 30 minutes"
    - "Remind me about the meeting in 15 minutes"
    """
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The reminder message (e.g., 'call mom', 'take medicine')",
            },
            "delay_seconds": {
                "type": "number",
                "description": "Delay in seconds before the reminder (e.g., 3600 for 1 hour, 60 for 1 minute)",
            },
        },
        "required": ["message", "delay_seconds"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Create a reminder."""
        message = kwargs.get("message", "").strip()
        delay = kwargs.get("delay_seconds")
        
        if not message:
            return {"error": "Message is required"}
        
        if not delay or delay <= 0:
            return {"error": "Delay must be a positive number"}
        
        delay = float(delay)
        
        # Get task manager from deps
        if not hasattr(deps, 'task_manager') or deps.task_manager is None:
            return {"error": "Task manager not available"}
        
        task_id = deps.task_manager.create_reminder(message, int(delay))
        
        # Format delay nicely
        if delay < 60:
            delay_str = f"{int(delay)} seconds"
        elif delay < 3600:
            minutes = int(delay // 60)
            delay_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = int(delay // 3600)
            minutes = int((delay % 3600) // 60)
            if minutes > 0:
                delay_str = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
            else:
                delay_str = f"{hours} hour{'s' if hours != 1 else ''}"
        
        logger.info(f"Tool call: create_reminder message='{message}' delay={delay_str}")
        
        return {
            "task_id": task_id,
            "message": message,
            "delay": delay_str,
            "status": "Reminder set"
        }