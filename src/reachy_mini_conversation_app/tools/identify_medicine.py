import logging
import base64
import io
import asyncio
from typing import Any, Dict
from PIL import Image
import cv2
import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class IdentifyMedicine(Tool):
    """Identify medicine, medication, pills, or prescriptions from camera."""

    name = "identify_medicine"
    description = """Identify medicine, medication, pills, tablets, or prescription bottles from camera. 
    Use this tool when the user asks about medicine, medication, pills, prescriptions, or shows you a pill bottle.
    Keywords: medicine, medication, pills, prescription, drug, tablet, capsule"""
    
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identify medicine from camera."""
        logger.info("Tool call: identify_medicine")

        # Get frame from camera
        if deps.camera_worker is None:
            logger.error("Camera worker not available")
            return {"error": "Camera not available"}
        
        # Add delay for user to position medicine
        logger.info("Waiting 2 seconds for user to position medicine...")
        await asyncio.sleep(2.0)
        
        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            logger.error("No camera frame available")
            return {"error": "No camera frame available"}

        logger.info(f"Got frame with shape: {frame.shape}")

        # Convert to base64 for Ollama
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            logger.info(f"Image encoded, base64 length: {len(img_str)}")
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return {"error": f"Image encoding failed: {e}"}

        # Simple prompt
        prompt = """Look at this image. If you see any medicine bottles, pills, or medication packaging:
1. Identify the medicine name if visible
2. Note any dosage information  
3. Mention any timing instructions (morning, evening, with food, etc.)

If you don't see any medicine, just say what you do see."""

        try:
            logger.info("Sending request to Ollama...")
            
            # Use a reasonable timeout
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "gemma3:4b",
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [img_str]
                            }
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 100,
                        }
                    }
                )

                logger.info(f"Ollama response status: {response.status_code}")
                
                result = response.json()
                medicine_info = result["message"]["content"].strip()
                
                logger.info(f"Medicine identified: {medicine_info}")
                
                return {"medicine_info": medicine_info}
                
        except httpx.TimeoutException:
            logger.error("Ollama request timed out after 30 seconds")
            return {"error": "Vision request timed out - Ollama may be processing slowly"}
        except Exception as e:
            logger.error(f"Medicine identification error: {e}", exc_info=True)
            return {"error": f"Failed to identify medicine: {str(e)}"}