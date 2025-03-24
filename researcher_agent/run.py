from dotenv import load_dotenv
import json
import logging
import os
from researcher_agent.schemas import InputSchema, ResearchOutput
from typing import Dict, List, Any
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.inference import InferenceClient
from naptha_sdk.user import sign_consumer_id

load_dotenv()

logger = logging.getLogger(__name__)

class ResearcherAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.node = InferenceClient(self.deployment.node)

    async def create(self, deployment: AgentDeployment):
        """Create the agent with the given deployment configuration."""
        self.deployment = deployment
        self.node = InferenceClient(self.deployment.node)
        return self

    def _clean_text(self, text: str) -> str:
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

    def _extract_findings(self, content: str) -> List[Dict[str, Any]]:
        findings = []
        current_section = None
        current_points = []
        in_findings_section = False
        in_metadata_section = False
        in_questions_section = False

        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith(('**metadata**', '### metadata', '# metadata')):
                in_metadata_section = True
                in_findings_section = False
                in_questions_section = False
                continue
            elif line.lower().startswith(('**questions**', '### questions', '# questions')):
                in_questions_section = True
                in_findings_section = False
                in_metadata_section = False
                continue
            elif line.lower().startswith(('**findings**', '### findings', '# findings')):
                in_findings_section = True
                in_metadata_section = False
                in_questions_section = False
                continue

            if in_metadata_section or in_questions_section:
                continue

            if not in_findings_section and not any(line.lower().startswith(('**', '###', '#')) for line in lines[i:i+5]):
                in_findings_section = True

            if not in_findings_section:
                continue

            if line[0].isdigit() and '. ' in line:
                if current_section and current_points:
                    findings.append({
                        "section": current_section,
                        "points": current_points.copy()
                    })
                section_title = line.split('. ', 1)[1].strip()
                section_title = section_title.replace('**', '').strip()
                current_section = section_title
                current_points = []
            elif line.startswith('**') and line.endswith('**'):
                if current_section and current_points:
                    findings.append({
                        "section": current_section,
                        "points": current_points.copy()
                    })
                current_section = self._clean_text(line)
                current_points = []
            elif line.startswith('###'):
                if current_section and current_points:
                    findings.append({
                        "section": current_section,
                        "points": current_points.copy()
                    })
                current_section = line.replace('###', '').strip()
                current_points = []
            elif line.startswith('#'):
                if current_section and current_points:
                    findings.append({
                        "section": current_section,
                        "points": current_points.copy()
                    })
                current_section = line.lstrip('#').strip()
                current_points = []
            elif line.endswith(':') and not line.startswith('-'):
                if current_section and current_points:
                    findings.append({
                        "section": current_section,
                        "points": current_points.copy()
                    })
                current_section = line[:-1].strip()
                current_points = []
            elif line.startswith('- '):
                point = self._clean_text(line)
                if point and point not in current_points:
                    current_points.append(point)
            elif line.startswith('  ') or line.startswith('\t'):
                point = self._clean_text(line)
                if point and point not in current_points:
                    current_points.append(point)
            elif current_section and line:
                point = self._clean_text(line)
                if point and point not in current_points:
                    current_points.append(point)
            elif not current_section:
                current_section = "Key Findings"

        if current_section and current_points:
            findings.append({
                "section": current_section,
                "points": current_points
            })

        if not findings:
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if paragraphs:
                findings.append({
                    "section": "Key Findings",
                    "points": [self._clean_text(p) for p in paragraphs]
                })

        cleaned_findings = []
        for finding in findings:
            section = finding["section"]
            points = finding["points"]
            
            if section.lower() in ["metadata", "questions", "summary"]:
                continue
                
            cleaned_points = []
            for point in points:
                if not any(x in point.lower() for x in ["metadata:", "questions:", "research focus:", "key sources:", "analysis approach:"]):
                    cleaned_points.append(point)
            
            if cleaned_points:
                cleaned_findings.append({
                    "section": section,
                    "points": cleaned_points
                })

        return cleaned_findings

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
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
                        value = self._clean_text(''.join(parts[2:]))
                        if value.startswith(':'):
                            value = value[1:].strip()
                        metadata[key] = value
                elif line.startswith('- '):
                    parts = line[2:].split(':', 1)
                    if len(parts) == 2:
                        key = self._clean_text(parts[0])
                        value = self._clean_text(parts[1])
                        metadata[key] = value
        
        return metadata

    async def research(self, inputs: InputSchema) -> Dict[str, Any]:
        """Run research on the given topic."""
        system_prompt = self.deployment.config.system_prompt.get('role', '') if isinstance(self.deployment.config.system_prompt, dict) else str(self.deployment.config.system_prompt)
        
        questions_to_answer = []
        if inputs.context and "previous_questions" in inputs.context:
            questions_to_answer.extend(inputs.context["previous_questions"])
        
        questions_to_answer = [self._clean_text(q) for q in questions_to_answer]
        questions_to_answer = list(dict.fromkeys(questions_to_answer))
        
        messages = [{"role": "system", "content": system_prompt}]
        
        messages.append({"role": "user", "content": inputs.topic})

        if questions_to_answer:
            question_prompt = (
                "Your primary task is to answer these specific questions: "
                f"{chr(10).join(f'- {q}' for q in questions_to_answer)}\n\n"
                "Please provide a focused response that: "
                "1. Provides a fresh perspective and new insights "
                "2. Includes concrete examples and evidence "
                "3. Addresses potential challenges or limitations "
                "4. Suggests practical applications and solutions "
                "5. Generates new follow-up questions for further research\n\n"
                "IMPORTANT: Structure your findings as follows:\n"
                "1. Create a separate section for each question you're answering\n"
                "2. Under each section, provide specific points that address that question\n"
                "3. Format each section as:\n"
                "   Section: [Question being answered]\n"
                "   Points:\n"
                "   - [Point 1]\n"
                "   - [Point 2]\n"
                "   etc.\n"
                "4. Ensure each point directly relates to the section's question"
            )

            system_prompt = f"{system_prompt}\n\n{question_prompt}"

        if inputs.context and "relevant_findings" in inputs.context:
            relevant_findings = inputs.context["relevant_findings"]
            if relevant_findings:
                findings_prompt = (
                    "Here are some relevant findings from previous research that may help inform your response:\n"
                    f"{chr(10).join(f'- {finding}' for finding in relevant_findings)}\n\n"
                    "Please consider these findings while providing your own unique perspective and insights."
                )
                system_prompt = f"{system_prompt}\n\n{findings_prompt}"

        response = await self.node.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": messages,
            "temperature": inputs.temperature or self.deployment.config.llm_config.temperature,
            "max_tokens": inputs.max_tokens or self.deployment.config.llm_config.max_tokens
        })

        if isinstance(response, dict):
            content = response['choices'][0]['message']['content']
        else:
            content = response.choices[0].message.content
        
        findings = self._extract_findings(content)
        metadata = self._extract_metadata(content)

        if not findings:
            findings = [{
                "section": "Key Findings",
                "points": [self._clean_text(p) for p in content.split('\n\n') if p.strip()]
            }]

        clean_questions = [self._clean_text(q) for q in questions_to_answer]
        questions_text = " ".join(clean_questions)
        findings_text = " ".join(f"{finding.get('section', '')}: {', '.join(finding.get('points', []))}"
                               for finding in findings)
        summary_prompt = (
            f"Based on the following research findings about {inputs.topic}, "
            "generate a clear, concise summary in 2-3 sentences. "
            "The summary should: "
            "1. Focus only on the most important findings "
            "2. Avoid any special characters or quotes "
            "3. Be written in plain, clean text "
            "4. Not include any markdown formatting "
            "5. Mention which questions were answered "
            f"Questions being addressed: {questions_text} "
            f"Research Findings: {findings_text}"
        )

        summary_messages = [
            {"role": "system", "content": "You are a research assistant that generates clean, concise summaries. Use only plain text, no special characters or formatting."},
            {"role": "user", "content": summary_prompt}
        ]

        summary_response = await self.node.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": summary_messages,
            "temperature": 0.7,
            "max_tokens": 200
        })

        if isinstance(summary_response, dict):
            summary = summary_response['choices'][0]['message']['content']
        else:
            summary = summary_response.choices[0].message.content

        summary = self._clean_text(summary)

        questions_prompt = (
            f"Based on the research findings about {inputs.topic}, "
            "generate 3-5 focused research questions that would help deepen our understanding. "
            "Each question should be specific, actionable, and explore important aspects "
            "that need further investigation. Format each question on a new line starting with a dash (-)."
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": questions_prompt})

        questions_response = await self.node.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        })

        if isinstance(questions_response, dict):
            questions_content = questions_response['choices'][0]['message']['content']
        else:
            questions_content = questions_response.choices[0].message.content

        # Extract questions from the response and clean them
        questions = [
            self._clean_text(q.lstrip('- ').strip()) 
            for q in questions_content.split('\n') 
            if q.strip().startswith('-')
        ]

        metadata["questions_being_addressed"] = [self._clean_text(q) for q in (questions_to_answer if questions_to_answer else [])]

        output = ResearchOutput(
            findings=findings,
            questions=questions,
            metadata=metadata,
            summary=summary
        ).model_dump()

        return output

async def run(module_run: Dict, *args, **kwargs):
    """Main entry point for the module."""
    try:
        run_input = AgentRunInput(**module_run)
        run_input.inputs = InputSchema(**run_input.inputs)
        researcher = ResearcherAgent(run_input.deployment)
        result = await researcher.research(run_input.inputs)
        return result
    except Exception as e:
        logger.error(f"Error in researcher_agent run: {e}")
        raise e

if __name__ == "__main__":
    import asyncio
    import json
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    naptha = Naptha()

    async def test_researcher():
        deployment = await setup_module_deployment(
            "agent",
            "researcher_agent/configs/deployment.json",
            node_url=os.getenv("NODE_URL")
        )

        test_run_input = {
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY")),
            "inputs": {
                "topic": "AI in healthcare"
            }
        }

        result = await run(test_run_input)
        print("Run method result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_researcher())