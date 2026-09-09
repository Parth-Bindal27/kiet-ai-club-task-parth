"""Prompt templates for generation."""

PROMPT_TEMPLATES = {
    "alpaca": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
}

def format_prompt(instruction: str, input_text: str = "", template: str = "alpaca") -> str:
    """Formats a prompt using the specified template."""
    tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["alpaca"])
    return tmpl.format(instruction=instruction, input=input_text)

def extract_response(full_output: str, template: str = "alpaca") -> str:
    """Extracts just the response part from the model's full generation."""
    if template == "alpaca":
        marker = "### Response:\n"
        if marker in full_output:
            return full_output.split(marker)[-1].strip()
    return full_output.strip()
