import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class MemoryStore:
    """Store and retrieve memory of clip extraction performance and user feedback."""
    
    def __init__(self, memory_file: str = "clip_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
    
    def _load_memory(self) -> Dict:
        """Load memory from file or create new if not exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"clips": {}, "metadata": {"created_at": datetime.now().isoformat()}}
        else:
            return {"clips": {}, "metadata": {"created_at": datetime.now().isoformat()}}
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def add_memory(self, clip_id: str, data: Dict[str, Any]) -> None:
        """Add a new clip memory with timestamp."""
        self.memory["clips"][clip_id] = {
            **data,
            "timestamp": datetime.now().isoformat(),
            "feedback": None
        }
        self._save_memory()
    
    def add_feedback(self, clip_id: str, is_satisfied: bool, 
                   feedback_text: Optional[str] = None) -> bool:
        """Add user feedback for a specific clip."""
        if clip_id not in self.memory["clips"]:
            return False
        
        self.memory["clips"][clip_id]["feedback"] = {
            "is_satisfied": is_satisfied,
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_memory()
        return True
    
    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Get a clip by ID."""
        return self.memory["clips"].get(clip_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory performance."""
        total = len(self.memory["clips"])
        if total == 0:
            return {
                "total_clips": 0,
                "feedback_received": 0,
                "satisfied_percentage": 0,
                "unsatisfied_percentage": 0
            }
        
        with_feedback = 0
        satisfied = 0
        
        for clip_id, data in self.memory["clips"].items():
            if data.get("feedback"):
                with_feedback += 1
                if data["feedback"].get("is_satisfied"):
                    satisfied += 1
        
        return {
            "total_clips": total,
            "feedback_received": with_feedback,
            "satisfied_percentage": (satisfied / with_feedback * 100) if with_feedback > 0 else 0,
            "unsatisfied_percentage": ((with_feedback - satisfied) / with_feedback * 100) if with_feedback > 0 else 0
        }