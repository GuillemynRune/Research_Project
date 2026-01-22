"""Entrypoint for the Reachy Mini conversation app - Modified for local models support."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Import config to check which handler to use
    from reachy_mini_conversation_app.config import config
    
    # Putting these dependencies here makes the dashboard faster to load
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
    from reachy_mini_conversation_app.task_manager import TaskManager

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    
    # Show which mode we're using
    if config.USE_LOCAL_MODELS:
        logger.info("=" * 60)
        logger.info("USING LOCAL MODELS")
        logger.info(f"  Whisper: {config.WHISPER_MODEL}")
        logger.info(f"  Gemma 3 VLM: {config.GEMMA_MODEL}")
        logger.info(f"  (Vision-Language Model - handles both text and images)")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("USING OPENAI API")
        logger.info(f"  Model: {config.MODEL_NAME}")
        logger.info("=" * 60)

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Check if running in simulation mode without --gradio
    if robot.client.get_status()["simulation_enabled"] and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode."
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
        task_manager=None,
    )
    
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    # Choose handler based on configuration
    if config.USE_LOCAL_MODELS:
        logger.info("Initializing LOCAL conversation handler")
        from local_conversation_handler import LocalConversationHandler
        handler = LocalConversationHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)
        # Head wobbler not needed for local models (no streaming audio from LLM)
        head_wobbler.stop()
    else:
        logger.info("Initializing OPENAI conversation handler")
        from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
        handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        # === CAMERA MONITOR INTERFACE (No audio streaming) ===
        import cv2
        import numpy as np
        
        with gr.Blocks(title="Reachy Mini Camera Monitor") as enhanced_ui:
            gr.Markdown("# 🤖 Reachy Mini - Camera Monitor")
            
            with gr.Row():
                # Camera feed (larger)
                with gr.Column(scale=2):
                    gr.Markdown("### 📹 Camera Feed")
                    camera_display = gr.Image(type="numpy", height=480, interactive=False, show_label=False)
                
                # Conversation history
                with gr.Column(scale=1):
                    gr.Markdown("### 💬 Conversation History")
                    history_display = gr.Textbox(lines=20, max_lines=20, interactive=False, show_label=False)
                    
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            def get_camera():
                if deps.camera_worker is None:
                    return None
                frame = deps.camera_worker.get_latest_frame()
                if frame is None:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            def get_history():
                if not hasattr(handler, 'conversation_history'):
                    return "No history available"
                history = handler.conversation_history[-10:]  # Show last 10 messages
                lines = []
                for msg in history:
                    role = msg.get('role', '?')
                    content = str(msg.get('content', ''))
                    lines.append(f"{role}: {content}\n")
                return "\n".join(lines) if lines else "No messages yet"
            
            def update_monitors():
                return get_camera(), get_history()
            
            # Manual refresh button
            refresh_btn.click(fn=update_monitors, outputs=[camera_display, history_display])
            
            # Auto-refresh using timer (0.1 seconds = 10 FPS for smooth camera feed)
            timer = gr.Timer(value=0.1, active=True)
            timer.tick(fn=update_monitors, outputs=[camera_display, history_display])
        
        stream_manager = enhanced_ui
        # === END CAMERA MONITOR INTERFACE ===
        
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        app = gr.mount_gradio_app(app, stream_manager, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )
        stream_manager.deps = deps

    # Start async services
    movement_manager.start()
    if not config.USE_LOCAL_MODELS:  # Only use head wobbler with OpenAI streaming
        head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        if not config.USE_LOCAL_MODELS:
            head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()