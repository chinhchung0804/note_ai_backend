import mimetypes
import os

def detect_input_type(file_path: str) -> str:
    """
    Detect loại input file
    Returns: 'image', 'audio', 'pdf', 'docx', 'doc', hoặc 'text'
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return 'pdf'
    if ext in ['.docx', '.doc']:
        return 'docx'
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return 'text'
    
    if mime_type.startswith('image'):
        return 'image'
    if mime_type.startswith('audio'):
        return 'audio'
    if mime_type == 'application/pdf':
        return 'pdf'
    if mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                     'application/msword']:
        return 'docx'
    
    return 'text'
