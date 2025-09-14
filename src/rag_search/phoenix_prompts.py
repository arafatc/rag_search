import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Phoenix integration with fallback
try:
    import phoenix as px
    from phoenix.client.resources.prompts import PromptVersion, v1
    from phoenix.trace import using_project
    PHOENIX_AVAILABLE = True
    print("INFO: Phoenix integration available for prompt management")
except ImportError as e:
    PHOENIX_AVAILABLE = False
    print(f"WARNING: Phoenix integration not available - {e}")

class PhoenixPrompts:
    """Minimal Phoenix prompt manager with fallback"""
    
    def __init__(self):
        self.local_prompts = self._load_local_prompts()
        self.phoenix_client = self._init_phoenix() if PHOENIX_AVAILABLE else None
        self.usage_stats = {}  # Track prompt usage for observability
    
    def _load_local_prompts(self):
        """Load prompts from JSON file"""
        try:
            prompts_file = Path(__file__).parent.parent / "prompts" / "prompts.json"
            with open(prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load prompts.json: {e}")
            return {}
    
    def _init_phoenix(self):
        """Initialize Phoenix client"""
        try:
            # Phoenix Client() connects to default endpoint (localhost:6006)
            client = px.Client()
            print("INFO: Phoenix client connected successfully")
            return client
        except Exception as e:
            print(f"Warning: Failed to initialize Phoenix client: {e}")
            return None
    
    def get_prompt(self, name: str, version: str = "latest") -> str:
        """Get prompt with Phoenix -> JSON -> ENV fallback and observability"""
        start_time = time.time()
        
        # Try Phoenix first
        if self.phoenix_client:
            try:
                # Fixed: Remove tag parameter for basic retrieval
                prompt_version = self.phoenix_client.prompts.get(
                    prompt_identifier=name
                )
                if prompt_version:
                    # Extract template from Phoenix prompt
                    template = self._extract_template_from_phoenix(prompt_version)
                    if template:
                        execution_time = time.time() - start_time
                        self._record_usage(name, "phoenix", execution_time)
                        print(f"INFO: Retrieved prompt '{name}' from Phoenix")
                        return template
            except Exception as e:
                print(f"Warning: Phoenix prompt retrieval failed for {name}: {e}")
        
        # Fallback to local JSON (this is working!)
        if name in self.local_prompts:
            execution_time = time.time() - start_time
            self._record_usage(name, "json", execution_time)
            print(f"INFO: Using JSON prompt for '{name}'")
            return self.local_prompts[name]
        
        # Final fallback to existing environment variables (no breaking changes!)
        env_value = os.getenv(f"PROMPT_{name.upper()}", "")
        if env_value:
            execution_time = time.time() - start_time
            self._record_usage(name, "env", execution_time)
            print(f"INFO: Using environment variable prompt for '{name}'")
        return env_value
    
    def _record_usage(self, name: str, source: str, execution_time: float):
        """Record prompt usage for observability"""
        if name not in self.usage_stats:
            self.usage_stats[name] = {
                'total_calls': 0,
                'phoenix_calls': 0,
                'json_calls': 0,
                'env_calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'last_accessed': None
            }
        
        stats = self.usage_stats[name]
        stats['total_calls'] += 1
        stats[f'{source}_calls'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_calls']
        stats['last_accessed'] = datetime.now().isoformat()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive prompt usage statistics"""
        return {
            'individual_prompts': self.usage_stats,
            'summary': {
                'total_prompts': len(self.usage_stats),
                'total_calls': sum(s['total_calls'] for s in self.usage_stats.values()),
                'phoenix_integration_rate': self._calculate_phoenix_rate(),
                'average_response_time': self._calculate_avg_response_time()
            }
        }
    
    def _calculate_phoenix_rate(self) -> float:
        """Calculate percentage of calls served by Phoenix vs fallbacks"""
        if not self.usage_stats:
            return 0.0
        
        total_calls = sum(s['total_calls'] for s in self.usage_stats.values())
        phoenix_calls = sum(s['phoenix_calls'] for s in self.usage_stats.values())
        
        return (phoenix_calls / total_calls) * 100 if total_calls > 0 else 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate overall average response time"""
        if not self.usage_stats:
            return 0.0
        
        total_time = sum(s['total_time'] for s in self.usage_stats.values())
        total_calls = sum(s['total_calls'] for s in self.usage_stats.values())
        
        return total_time / total_calls if total_calls > 0 else 0.0
    
    def _extract_template_from_phoenix(self, prompt_version) -> Optional[str]:
        """Extract template string from Phoenix PromptVersion object"""
        try:
            # Method 1: Use the format() method to get OpenAI-style prompt
            formatted_prompt = prompt_version.format()
            if hasattr(formatted_prompt, 'messages') and formatted_prompt.messages:
                # Extract content from the first message (typically system message)
                first_message = formatted_prompt.messages[0]
                if isinstance(first_message, dict) and 'content' in first_message:
                    return first_message['content']
                elif hasattr(first_message, 'content'):
                    return first_message.content
            
            # Method 2: Access the internal _template structure
            if hasattr(prompt_version, '_template') and prompt_version._template:
                template_data = prompt_version._template
                if isinstance(template_data, dict) and 'messages' in template_data:
                    messages = template_data['messages']
                    if messages and len(messages) > 0:
                        first_message = messages[0]
                        if isinstance(first_message, dict) and 'content' in first_message:
                            return first_message['content']
            
            # Method 3: Fallback - try accessing prompt attribute directly
            if hasattr(prompt_version, 'prompt') and prompt_version.prompt:
                messages = prompt_version.prompt
                if messages and len(messages) > 0:
                    first_message = messages[0]
                    if isinstance(first_message, dict) and 'content' in first_message:
                        return first_message['content']
                    elif hasattr(first_message, 'content'):
                        return first_message.content
            
            return None
        except Exception as e:
            print(f"Warning: Failed to extract template from Phoenix prompt: {e}")
            return None
    
    def create_prompt_version(self, name: str, template: str, model_name: str = "gpt-3.5-turbo", description: str = "") -> bool:
        """Create or update prompt in Phoenix"""
        if not self.phoenix_client:
            print("Warning: Phoenix client not available")
            return False
        
        try:
            # Create PromptMessage objects for Phoenix (sequence required)
            messages = [
                v1.PromptMessage(
                    role="system",
                    content=template
                )
            ]
            
            # Create PromptVersion object with correct signature
            prompt_version = PromptVersion(
                messages,  # positional argument
                model_name=model_name,
                description=description,
                model_provider="OPENAI",
                template_format="F_STRING"
            )
            
            # Create the prompt in Phoenix
            created_prompt = self.phoenix_client.prompts.create(
                version=prompt_version,
                name=name,
                prompt_description=description
            )
            
            print(f"SUCCESS: Created prompt '{name}' in Phoenix with ID: {created_prompt.id}")
            return True
            
        except Exception as e:
            print(f"Error creating prompt '{name}' in Phoenix: {e}")
            return False

    def initialize_prompts(self):
        """Initialize prompts in Phoenix - Now with full Phoenix integration"""
        if not self.phoenix_client:
            print("Phoenix client not available for prompt initialization")
            return False
        
        print("INFO: Initializing prompts in Phoenix...")
        success_count = 0
        
        # Upload all JSON prompts to Phoenix
        for name, template in self.local_prompts.items():
            print(f"Creating prompt '{name}' in Phoenix...")
            if self.create_prompt_version(
                name=name,
                template=template,
                description=f"RAG System prompt for {name.replace('_', ' ').title()}"
            ):
                success_count += 1
        
        total_prompts = len(self.local_prompts)
        print(f"INFO: Successfully created {success_count}/{total_prompts} prompts in Phoenix")
        
        if success_count == total_prompts:
            print("SUCCESS: All prompts successfully uploaded to Phoenix!")
            return True
        else:
            print("WARNING: Some prompts failed to upload. JSON fallback remains available.")
            return False

# Global instance
_phoenix_prompts = PhoenixPrompts()

def get_prompt(name: str, version: str = "latest") -> str:
    """Simple function to get prompt - drop-in replacement with version support"""
    return _phoenix_prompts.get_prompt(name, version)

def create_prompt_version(name: str, template: str, model_name: str = "gpt-3.5-turbo", description: str = "") -> bool:
    """Create or update prompt in Phoenix"""
    return _phoenix_prompts.create_prompt_version(name, template, model_name, description)

def get_usage_statistics() -> Dict[str, Any]:
    """Get comprehensive usage statistics for observability"""
    return _phoenix_prompts.get_usage_stats()

def print_usage_report():
    """Print a formatted usage report for monitoring"""
    stats = get_usage_statistics()
    
    print("\n" + "="*60)
    print("PHOENIX PROMPT MANAGEMENT - USAGE REPORT")
    print("="*60)
    
    summary = stats['summary']
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"Total Calls: {summary['total_calls']}")
    print(f"Phoenix Integration Rate: {summary['phoenix_integration_rate']:.1f}%")
    print(f"Average Response Time: {summary['average_response_time']:.4f}s")
    
    print(f"\nIndividual Prompt Usage:")
    print("-" * 60)
    
    for name, stats in stats['individual_prompts'].items():
        print(f"  {name}:")
        print(f"    Total: {stats['total_calls']} calls")
        print(f"    Phoenix: {stats['phoenix_calls']} | JSON: {stats['json_calls']} | ENV: {stats['env_calls']}")
        print(f"    Avg Time: {stats['avg_time']:.4f}s")
        print(f"    Last Used: {stats['last_accessed']}")
        print()
    
    print("="*60 + "\n")

def initialize_phoenix_prompts():
    """Initialize all prompts in Phoenix"""
    return _phoenix_prompts.initialize_prompts()
