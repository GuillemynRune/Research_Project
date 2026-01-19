# src/reachy_mini_conversation_app/tools/identify_medicine.py
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class IdentifyMedicine(Tool):
    """Identify medicine from camera and set reminders."""

    name = "identify_medicine"
    description = "Look at medicine bottles/packages and help set up medication reminders"
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["identify", "set_schedule"],
                "description": "Whether to identify medicine or set a schedule"
            }
        },
        "required": ["action"]
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identify medicine and optionally set reminders."""
        import asyncio
        import cv2
        from PIL import Image
        
        action = kwargs.get("action", "identify")
        logger.info(f"Tool call: identify_medicine action={action}")

        # Get frame from camera
        if deps.camera_worker is None:
            return {"error": "Camera not available"}
        
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No camera frame available"}

        # Convert to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        # Use vision manager to analyze
        if deps.vision_manager is not None:
            prompt = """Look at this image carefully. If you see any medicine bottles, pills, or medication packaging:
1. Identify the medicine name if visible
2. Note any dosage information
3. Mention any timing instructions (morning, evening, with food, etc.)

If you don't see any medicine, just say what you do see."""
            
            result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image,
                frame,
                prompt
            )
            
            return {
                "medicine_info": result,
                "suggestion": "Would you like me to set up a reminder for this medication?"
            }
        
        return {"error": "Vision system not available"}