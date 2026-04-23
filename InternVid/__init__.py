"""
InternVid package for ViCLIP functionality
"""

# Import key modules
from .viclip import (
    get_viclip,
    retrieve_text,
    _frame_from_video,
    frames2tensor,
    get_vid_feat,
    get_text_feat_dict
)

__version__ = "1.0.0"
__all__ = [
    "get_viclip",
    "retrieve_text",
    "_frame_from_video",
    "frames2tensor",
    "get_vid_feat",
    "get_text_feat_dict"
]
