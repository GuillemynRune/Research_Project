"""Task Manager for handling reminders and timers.

This module provides asynchronous task scheduling for reminders and timers
that integrate with the conversation system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a scheduled task (reminder or timer)."""
    task_id: str
    task_type: str  # "reminder" or "timer"
    message: str
    scheduled_time: datetime
    created_time: datetime
    callback: Optional[Callable] = None
    is_completed: bool = False


class TaskManager:
    """Manages scheduled tasks like reminders and timers."""
    
    def __init__(self, tts_callback: Optional[Callable] = None):
        """Initialize task manager.
        
        Args:
            tts_callback: Optional callback function to speak notifications
                         Should accept a string message as parameter
        """
        self.tasks: Dict[str, Task] = {}
        self.tts_callback = tts_callback
        self._task_counter = 0
        self._running = False
        self._check_task = None
        logger.info("TaskManager initialized")
    
    async def start(self):
        """Start the task manager background loop."""
        if self._running:
            logger.warning("TaskManager already running")
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._check_tasks_loop())
        logger.info("TaskManager started")
    
    async def stop(self):
        """Stop the task manager."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskManager stopped")
    
    async def _check_tasks_loop(self):
        """Background loop to check for tasks that need to be triggered."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Check all tasks
                tasks_to_complete = []
                for task_id, task in self.tasks.items():
                    if not task.is_completed and current_time >= task.scheduled_time:
                        tasks_to_complete.append(task_id)
                
                # Complete tasks that are due
                for task_id in tasks_to_complete:
                    await self._complete_task(task_id)
                
                # Check every second
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task check loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _complete_task(self, task_id: str):
        """Complete a task by triggering its notification.
        
        Args:
            task_id: ID of the task to complete
        """
        task = self.tasks.get(task_id)
        if not task or task.is_completed:
            return
        
        completion_time = datetime.now()
        actual_elapsed = (completion_time - task.created_time).total_seconds()
        expected_elapsed = (task.scheduled_time - task.created_time).total_seconds()
        accuracy_diff = actual_elapsed - expected_elapsed

        logger.info(f"Completing task: {task_id} - {task.message}")
        logger.info(f"⏱️  Timer accuracy: expected={expected_elapsed:.2f}s, actual={actual_elapsed:.2f}s, diff={accuracy_diff:+.2f}s")
        task.is_completed = True
        
        # Speak the notification
        notification = f"Reminder: {task.message}" if task.task_type == "reminder" else f"Timer finished: {task.message}"
        
        if self.tts_callback:
            try:
                await self.tts_callback(notification)
            except Exception as e:
                logger.error(f"Error speaking notification: {e}")
        else:
            logger.info(f"[NOTIFICATION] {notification}")
        
        # Call custom callback if provided
        if task.callback:
            try:
                await task.callback(task)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
    
    def create_reminder(
        self, 
        message: str, 
        delay_seconds: int,
        callback: Optional[Callable] = None
    ) -> str:
        """Create a reminder that triggers after a delay.
        
        Args:
            message: Reminder message to speak
            delay_seconds: Seconds from now to trigger the reminder
            callback: Optional callback function to call when triggered
        
        Returns:
            Task ID of the created reminder
        
        Example:
            >>> task_id = task_manager.create_reminder("Take medicine", 3600)
            >>> # Will speak "Reminder: Take medicine" in 1 hour
        """
        self._task_counter += 1
        task_id = f"reminder_{self._task_counter}"
        
        created_time = datetime.now()
        scheduled_time = datetime.now() + timedelta(seconds=delay_seconds)
        
        task = Task(
            task_id=task_id,
            task_type="reminder",
            message=message,
            scheduled_time=scheduled_time,
            created_time=created_time,
            callback=callback
        )
        
        self.tasks[task_id] = task
        
        logger.info(
            f"Created reminder: {task_id} - '{message}' "
            f"scheduled for {scheduled_time.strftime('%H:%M:%S')}"
        )
        
        return task_id
    
    def create_timer(
        self,
        duration_seconds: int,
        callback: Optional[Callable] = None
    ) -> str:
        """Create a timer that triggers after a duration.
        
        Args:
            duration_seconds: Timer duration in seconds
            callback: Optional callback function to call when triggered
        
        Returns:
            Task ID of the created timer
        
        Example:
            >>> task_id = task_manager.create_timer(300)
            >>> # Will speak "Timer finished" in 5 minutes
        """
        self._task_counter += 1
        task_id = f"timer_{self._task_counter}"
        
        created_time = datetime.now()
        scheduled_time = datetime.now() + timedelta(seconds=duration_seconds)
        
        # Format duration nicely
        if duration_seconds < 60:
            duration_str = f"{duration_seconds} seconds"
        elif duration_seconds < 3600:
            minutes = duration_seconds // 60
            duration_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
        else:
            hours = duration_seconds // 3600
            minutes = (duration_seconds % 3600) // 60
            if minutes > 0:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
            else:
                duration_str = f"{hours} hour{'s' if hours != 1 else ''}"
        
        task = Task(
            task_id=task_id,
            task_type="timer",
            message=duration_str,
            scheduled_time=scheduled_time,
            created_time=created_time,
            callback=callback
        )
        
        self.tasks[task_id] = task
        
        logger.info(
            f"Created timer: {task_id} - {duration_str} "
            f"scheduled for {scheduled_time.strftime('%H:%M:%S')}"
        )
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task.
        
        Args:
            task_id: ID of the task to cancel
        
        Returns:
            True if task was cancelled, False if not found
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if not task.is_completed:
                task.is_completed = True
                logger.info(f"Cancelled task: {task_id}")
                return True
        return False
    
    def get_active_tasks(self) -> list[Task]:
        """Get all active (not completed) tasks.
        
        Returns:
            List of active tasks
        """
        return [
            task for task in self.tasks.values() 
            if not task.is_completed
        ]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: ID of the task
        
        Returns:
            Dictionary with task status or None if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        time_remaining = (task.scheduled_time - datetime.now()).total_seconds()
        
        return {
            "task_id": task.task_id,
            "type": task.task_type,
            "message": task.message,
            "scheduled_time": task.scheduled_time.isoformat(),
            "time_remaining_seconds": max(0, time_remaining),
            "is_completed": task.is_completed
        }


# Example usage
if __name__ == "__main__":
    async def test_task_manager():
        """Test the task manager."""
        
        async def speak_callback(message: str):
            """Mock TTS callback."""
            print(f"🔊 Speaking: {message}")
        
        # Initialize
        manager = TaskManager(tts_callback=speak_callback)
        await manager.start()
        
        print("TaskManager started")
        
        # Create a 3-second timer
        timer_id = manager.create_timer(3)
        print(f"Created timer: {timer_id}")
        
        # Create a 5-second reminder
        reminder_id = manager.create_reminder("Take your medicine", 5)
        print(f"Created reminder: {reminder_id}")
        
        # Wait for tasks to complete
        print("\nWaiting for tasks to trigger...")
        await asyncio.sleep(7)
        
        # Check status
        print("\nActive tasks:", manager.get_active_tasks())
        
        # Stop
        await manager.stop()
        print("\nTaskManager stopped")
    
    # Run test
    asyncio.run(test_task_manager())