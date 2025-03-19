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
        text = text.replace('**', '')
        text = text.lstrip('*- ')
        text = text.lstrip('0123456789. ')
        text = ' '.join(text.split())
        return text.strip()

    def _extract_findings(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured findings from the research content."""
        findings = []
        current_section = None
        current_points = []

        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('**') and line.endswith('**'):
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
            elif line[0].isdigit() and '. ' in line:
                point = line.split('. ', 1)[1].strip()
                point = self._clean_text(point)
                if point and point not in current_points:
                    current_points.append(point)
            elif line.startswith('-') or line.startswith('*'):
                point = self._clean_text(line)
                if point and point not in current_points:
                    current_points.append(point)

        if current_section and current_points:
            findings.append({
                "section": current_section,
                "points": current_points
            })

        return findings

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
        # First research pass to get findings and metadata
        messages = [{"role": "system", "content": json.dumps(self.deployment.config.system_prompt)}]
        messages.append({"role": "user", "content": inputs.topic})

        response = await self.node.run_inference({
            "model": self.deployment.config.llm_config.model,
            "messages": messages,
            "temperature": self.deployment.config.llm_config.temperature,
            "max_tokens": self.deployment.config.llm_config.max_tokens
        })

        if isinstance(response, dict):
            content = response['choices'][0]['message']['content']
        else:
            content = response.choices[0].message.content

        findings = self._extract_findings(content)
        metadata = self._extract_metadata(content)

        # Second pass to generate focused questions based on the findings
        questions_prompt = (
            f"Based on the research findings about {inputs.topic}, "
            "generate 3-5 focused research questions that would help deepen our understanding. "
            "Each question should be specific, actionable, and explore important aspects "
            "that need further investigation. Format each question on a new line starting with a dash (-)."
        )
        
        messages = [{"role": "system", "content": json.dumps(self.deployment.config.system_prompt)}]
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

        # Extract questions from the response
        questions = [
            q.lstrip('- ').strip() 
            for q in questions_content.split('\n') 
            if q.strip().startswith('-')
        ]

        return ResearchOutput(
            findings=findings,
            questions=questions,
            metadata=metadata
        ).model_dump()

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