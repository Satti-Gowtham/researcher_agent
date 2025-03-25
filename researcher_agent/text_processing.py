from typing import Dict, List, Any

def clean_text(text: str) -> str:
    """Clean text by removing markdown formatting."""
    if not text:
        return ""
        
    text = str(text)
    
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    
    text = text.lstrip('*- ')
    text = text.lstrip('0123456789. ')
    
    text = text.lstrip(' \t')
    
    text = ' '.join(text.split())
    
    text = text.replace('#', '')
    
    return text.strip()

def extract_findings(content: str) -> List[Dict[str, Any]]:
    """Extract findings from the content, handling various formats flexibly."""
    findings = []
    current_section = None
    current_points = []
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        if not line or line.lower().startswith(('**metadata**', '### metadata', '# metadata',
                                             '**questions**', '### questions', '# questions')):
            continue
        
        if line[0].isdigit() and '. ' in line:
            if current_section and current_points:
                findings.append({
                    "section": current_section,
                    "points": current_points.copy()
                })
            
            parts = line.split('. ', 1)
            if len(parts) == 2:
                section_title = parts[1].strip()
                section_title = clean_text(section_title)
                current_section = section_title
                current_points = []
                
                if ':' in section_title:
                    title_parts = section_title.split(':', 1)
                    if len(title_parts) == 2:
                        current_section = title_parts[0].strip()
                        current_points.append(title_parts[1].strip())
            continue
        
        if line.startswith(('- ', '* ', '  ', '\t')):
            point = clean_text(line)
            if point and point not in current_points:
                current_points.append(point)
            continue
        
        if current_section and line:
            if (line.startswith(('###', '#', '**')) or 
                (line.endswith(':') and not line.startswith('-')) or
                (line[0].isdigit() and '. ' in line)):
                continue
                
            point = clean_text(line)
            if point and point not in current_points:
                current_points.append(point)
        
        elif not current_section and len(line) > 20:
            current_section = clean_text(line)
            current_points = []
    
    if current_section and current_points:
        findings.append({
            "section": current_section,
            "points": current_points
        })
    
    if not findings:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            current_section = None
            current_points = []
            
            for para in paragraphs:
                lines = para.split('\n')
                first_line = lines[0].strip()
                
                if (len(first_line) > 20 and
                    not first_line.startswith(('-', '*', ' ')) and
                    not first_line.endswith('.')):
                    if current_section and current_points:
                        findings.append({
                            "section": current_section,
                            "points": current_points
                        })
                    current_section = clean_text(first_line)
                    current_points = []
                elif current_section:
                    current_points.append(clean_text(para))
                else:
                    current_section = "Key Findings"
                    current_points.append(clean_text(para))
            
            if current_section and current_points:
                findings.append({
                    "section": current_section,
                    "points": current_points
                })
    
    cleaned_findings = []
    for finding in findings:
        section = finding["section"]
        points = finding["points"]
        
        if section.lower() in ["metadata", "questions", "summary"]:
            continue
        
        cleaned_points = []
        for point in points:
            if not any(x in point.lower() for x in [
                "metadata:", "questions:", "research focus:", 
                "key sources:", "analysis approach:", "round:"
            ]):
                cleaned_points.append(point)
        
        if cleaned_points and len(cleaned_points) >= 1:
            cleaned_findings.append({
                "section": section,
                "points": cleaned_points
            })
    
    return cleaned_findings

def extract_metadata(content: str) -> Dict[str, Any]:
    """Extract metadata from the research content."""
    metadata = {}
    metadata_section = False
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('**Metadata**') or line.startswith('### Metadata'):
            metadata_section = True
            continue
            
        if metadata_section:
            if line.startswith('- **'):
                parts = line.split('**')
                if len(parts) >= 3:
                    key = parts[1].strip(':')
                    value = clean_text(''.join(parts[2:]))
                    if value.startswith(':'):
                        value = value[1:].strip()
                    metadata[key] = value
            elif line.startswith('- '):
                parts = line[2:].split(':', 1)
                if len(parts) == 2:
                    key = clean_text(parts[0])
                    value = clean_text(parts[1])
                    metadata[key] = value
    
    return metadata 