#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import difflib
import re
from datetime import datetime
import time

from openai import OpenAI


client: OpenAI | None = None


def call_openai(
    model: str,
    prompt: str,
    *,
    timeout: int | None = 3600,
    max_retries: int = 3,
):
    """Call the OpenAI chat completions API with retry and timeout.

    Parameters
    ----------
    model: str
        Name of the model to query.
    prompt: str
        The content to send as a user message.
    timeout: int | None, optional
        Maximum time to wait for a response in seconds (default: ``3600``).
        Pass ``None`` to wait indefinitely.
    max_retries: int, optional
        Number of times to retry the request on failure (default: 3).
    """
    if client is None:
        raise RuntimeError("OpenAI client has not been initialized")

    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            wait_msg = (
                f"Waiting up to {timeout} seconds" if timeout is not None else "Waiting indefinitely"
            )
            print(
                f"Calling OpenAI (attempt {attempt}/{max_retries})...\n{wait_msg} for {model} to respond."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # broad except to catch networking/timeout errors
            last_exception = exc
            if attempt < max_retries:
                sleep_time = 2 ** (attempt - 1)
                print(
                    f"OpenAI request failed: {exc}. Retrying in {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
    # If we reach here, all retries failed
    raise RuntimeError(
        f"OpenAI request failed after {max_retries} attempts: {last_exception}"
    ) from last_exception


def generate_idea(
    topic: str,
    field: str,
    question: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Generate a high-value research idea for the given topic, field and question."""
    prompt = (

        f"Provide a high-value, innovative and practical research idea for the topic: {topic}. "
        "Ensure the idea lends itself to rigorous methodology and clear exposition. "
        "Respond with a concise description."

    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def write_paper(
    topic: str,
    field: str,
    question: str,
    idea: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Draft a full research paper given a topic, field, research question and idea."""
    prompt = (
        "You are an expert researcher. Write a full research paper based on the following information.\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n"
        f"Research Question: {question}\n"
        f"Idea: {idea}\n\n"
 
        "The paper should include the following sections: Abstract, Introduction, Related Work, Methodology, "
        "Experiments, Results, Discussion, Conclusion, References.\n"
        "Use LaTeX formatting for a full-length paper, employing appropriate section commands. \n"
        "If code is necessary for the experiments, include it using the LaTeX lstlisting environment with "
        "language=Python.\n"
        "Ensure rigorous methodology and clear exposition throughout."
 
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def review_and_revise_paper(
    paper_content: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> tuple[str, str, bool]:
    """Review the paper and provide only the specific changes needed (diff format).
    
    Returns:
        tuple[str, str, bool]: (review_feedback, diff_changes, is_ready)
    """
    prompt = (
        "You are a top journal peer reviewer and editor. Your task is to:\n"
        "1. Review this research paper for novelty, clarity, methodology rigor, and significance\n"
        "2. Decide if it's ready for submission to a top journal\n"
        "3. If NOT ready, provide ONLY the specific changes needed\n\n"
        
        "Response format:\n"
        "=== EVALUATION ===\n"
        "READY FOR SUBMISSION: [YES/NO]\n\n"
        
        "=== REVIEW FEEDBACK ===\n"
        "[Provide constructive feedback on what needs improvement]\n\n"
        
        "=== CHANGES NEEDED ===\n"
        "CHANGE 1:\n"
        "LOCATION: [section name or approximate line]\n"
        "ORIGINAL: [exact text to replace]\n"
        "REVISED: [exact replacement text]\n\n"
        
        "CHANGE 2:\n"
        "LOCATION: [section name or approximate line]\n"
        "ORIGINAL: [exact text to replace]\n"
        "REVISED: [exact replacement text]\n\n"
        
        "[Continue for all changes...]\n\n"
        
        "IMPORTANT: Only output specific changes that need to be made. Do NOT include the entire paper.\n"
        "If ready for submission, simply state so in the evaluation section.\n\n"
        f"Paper to review:\n{paper_content}"
    )
    
    response = call_openai(model, prompt, timeout=timeout, max_retries=max_retries)
    
    # Parse the response to separate evaluation, feedback and changes
    is_ready = False
    feedback = ""
    changes = ""
    
    try:
        if "=== EVALUATION ===" in response:
            eval_section = response.split("=== EVALUATION ===")[1].split("=== REVIEW FEEDBACK ===")[0]
            is_ready = "YES" in eval_section.upper() and "READY FOR SUBMISSION" in eval_section.upper()
        
        if "=== REVIEW FEEDBACK ===" in response and "=== CHANGES NEEDED ===" in response:
            parts = response.split("=== REVIEW FEEDBACK ===")[1].split("=== CHANGES NEEDED ===")
            feedback = parts[0].strip()
            changes = parts[1].strip() if len(parts) > 1 else ""
        elif "=== REVIEW FEEDBACK ===" in response:
            feedback = response.split("=== REVIEW FEEDBACK ===")[1].strip()
        else:
            # Fallback if format is not followed
            feedback = response[:500] + "..." if len(response) > 500 else response
            
    except Exception as e:
        print(f"Warning: Could not parse response format: {e}")
        feedback = response
        is_ready = "ready for submission" in response.lower() or "yes" in response.lower()
    
    return feedback, changes, is_ready


def apply_diff_changes(paper_content: str, changes: str) -> str:
    """Apply diff-style changes to paper content."""
    if not changes or changes.strip() == "":
        return paper_content
        
    change_blocks = re.split(r'CHANGE \d+:', changes)
    modified_content = paper_content
    changes_applied = 0
    
    for block in change_blocks[1:]:  # Skip empty first split
        try:
            # Extract location, original, and revised text
            location_match = re.search(r'LOCATION:\s*(.*?)(?=\n)', block)
            original_match = re.search(r'ORIGINAL:\s*(.*?)\s*REVISED:', block, re.DOTALL)
            revised_match = re.search(r'REVISED:\s*(.*?)(?=\n\n|CHANGE|\Z)', block, re.DOTALL)
            
            if original_match and revised_match:
                original_text = original_match.group(1).strip()
                revised_text = revised_match.group(1).strip()
                location = location_match.group(1).strip() if location_match else "Unknown"
                
                # Apply the change (replace first occurrence)
                if original_text in modified_content:
                    modified_content = modified_content.replace(original_text, revised_text, 1)
                    changes_applied += 1
                    print(f"✓ Applied change {changes_applied} in {location}")
                else:
                    print(f"⚠ Warning: Could not find original text in {location}")
                    
        except Exception as e:
            print(f"⚠ Warning: Could not parse change block. Error: {e}")
            continue
    
    print(f"Applied {changes_applied} changes total.")
    return modified_content


def evaluate_paper(
    paper_content: str,
    feedback: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Decide if the paper is ready for submission.

    An editor response of ``NO`` must include clear guidance on how to
    improve the paper rather than rejecting it outright.
    """
    prompt = (
        "You are a journal editor evaluating a paper and the peer review feedback. Decide whether the paper "
        "is of top quality and can be submitted to a top journal without any modifications. Respond with 'YES' "
        "if it is ready. If it is not ready, respond with 'NO' and provide specific suggestions to improve the paper rather than rejecting it.\n\n"
        f"Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def revise_paper(
    paper_content: str,
    feedback: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Revise the paper based on review feedback."""
    prompt = (
        "Based on the reviewer feedback provided, revise the research paper to address all issues and improve its "
 
        "quality. Provide the complete revised paper as a LaTeX document using \\documentclass{article}. Use the "
        "lstlisting environment for any Python code blocks. Do not include any explanations, only the revised paper.\n\n"
 
        f"Original Paper:\n{paper_content}\n\n"
        f"Review Feedback:\n{feedback}"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def modify_existing_paper(
    existing_paper_content: str,
    topic: str,
    field: str,
    question: str,
    idea: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Modify an existing paper based on new topic, field, question, and idea."""
    prompt = (
        "You are an expert researcher. Modify the existing research paper to incorporate the new research direction "
        "while preserving the structure and quality of the original work. Maintain the LaTeX formatting and "
        "academic writing style.\n\n"
        f"New Research Direction:\n"
        f"Field: {field}\n"
        f"Topic: {topic}\n"
        f"Research Question: {question}\n"
        f"New Idea: {idea}\n\n"
        f"Existing Paper to Modify:\n{existing_paper_content}\n\n"
        "Instructions:\n"
        "- Keep the same LaTeX document structure and formatting\n"
        "- Update the title, abstract, and content to reflect the new research direction\n"
        "- Maintain academic rigor and clear exposition\n"
        "- If code is necessary, use the LaTeX lstlisting environment with language=Python\n"
        "- Ensure all sections are updated consistently with the new research focus\n"
        "- Provide only the complete modified paper, no explanations"
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def iteratively_improve_paper(
    existing_paper_content: str,
    model: str,
    *,
    timeout: int | None,
    max_retries: int,
) -> str:
    """Iteratively improve an existing paper without changing its core topic/field/question."""
    prompt = (
        "You are an expert researcher tasked with improving an existing research paper. "
        "Your goal is to enhance the paper's quality, clarity, rigor, and impact while "
        "keeping the same research topic, field, and core research question.\n\n"
        "Improvements you should focus on:\n"
        "- Strengthen the theoretical foundations and methodology\n"
        "- Improve clarity of exposition and writing quality\n"
        "- Add more rigorous experimental validation if applicable\n"
        "- Enhance the literature review and related work sections\n"
        "- Improve mathematical formulations and proofs\n"
        "- Add more comprehensive analysis and discussion\n"
        "- Strengthen the conclusion and future work sections\n"
        "- Fix any technical issues or gaps in reasoning\n\n"
        "Important constraints:\n"
        "- DO NOT change the core research topic, field, or main research question\n"
        "- DO NOT change the fundamental research direction or approach\n"
        "- Maintain the same LaTeX document structure and formatting\n"
        "- Keep all existing section organization\n"
        "- Preserve any working code in lstlisting environments\n\n"
        f"Existing Paper to Improve:\n{existing_paper_content}\n\n"
        "Provide only the complete improved paper as a LaTeX document, no explanations."
    )
    return call_openai(model, prompt, timeout=timeout, max_retries=max_retries)


def extract_paper_metadata(paper_content: str) -> tuple[str, str, str]:
    """Extract topic, field, and research question from existing paper content."""
    # Try to extract from title and abstract
    title_match = re.search(r'\\title\{([^}]+)\}', paper_content)
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', paper_content, re.DOTALL)
    
    title = title_match.group(1) if title_match else "Unknown Topic"
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # Extract key concepts from title and abstract to infer field and topic
    field = "Computer Science"  # Default field
    topic = title.split(':')[0] if ':' in title else title
    question = f"Research question derived from: {title}"
    
    # Try to be more specific based on content
    content_lower = (title + " " + abstract).lower()
    if any(word in content_lower for word in ['quantum', 'qubit', 'entangle']):
        field = "Quantum Computing"
    elif any(word in content_lower for word in ['neural', 'deep learning', 'machine learning', 'ai']):
        field = "Machine Learning"
    elif any(word in content_lower for word in ['biology', 'protein', 'gene', 'medical']):
        field = "Computational Biology"
    elif any(word in content_lower for word in ['security', 'crypto', 'attack', 'defense']):
        field = "Cybersecurity"
    
    return topic, field, question


def save_paper_and_code(paper_content: str, output_dir: Path | str, filename: str = "paper.tex") -> Path:
    """Save the paper and Python snippets to a unique subdirectory.

    A timestamped subfolder is created inside ``output_dir`` so multiple
    papers can coexist. If ``output_dir`` already points to an existing paper
    directory (containing any .tex file), files are updated in place. Python
    code blocks contained in LaTeX ``lstlisting`` environments are extracted to
    ``code_<n>.py`` files. The directory where the .tex and code files are
    written is returned for subsequent steps.
    """
    base_path = Path(output_dir)

    # Check if there's already a .tex file in the directory
    existing_tex = find_tex_file(base_path) if base_path.exists() else None
    if existing_tex:
        paper_dir = base_path
        paper_path = existing_tex  # Keep original filename
    else:
        base_path.mkdir(parents=True, exist_ok=True)
        paper_dir = base_path / datetime.now().strftime("%Y%m%d_%H%M%S")
        paper_dir.mkdir(parents=True, exist_ok=True)
        paper_path = paper_dir / filename

    paper_path.write_text(paper_content, encoding="utf-8")

    # extract python code blocks from LaTeX lstlisting environments
    code_blocks = re.findall(
        r"\\begin{lstlisting}(?:\[[^\]]*\])?\s*(.*?)\s*\\end{lstlisting}",
        paper_content,
        re.DOTALL,
    )

    for idx, code in enumerate(code_blocks, 1):
        code_file = paper_dir / f"code_{idx}.py"
        code_file.write_text(code.strip(), encoding="utf-8")
    return paper_dir


def apply_diff_and_save(original_path: Path, new_content: str) -> str:
    """Apply modifications to the original file and return a unified diff."""
    original = original_path.read_text(encoding="utf-8")
    diff = difflib.unified_diff(
        original.splitlines(),
        new_content.splitlines(),
        fromfile=str(original_path),
        tofile="revised_" + original_path.name,
        lineterm=""
    )
    diff_text = "\n".join(diff)
    # write revised content
    original_path.write_text(new_content, encoding="utf-8")
    return diff_text


def find_tex_file(output_dir: Path) -> Path | None:
    """Find any .tex file in the output directory."""
    tex_files = list(output_dir.glob("*.tex"))
    if tex_files:
        # Prefer paper.tex if it exists, otherwise return the first .tex file found
        paper_tex = output_dir / "paper.tex"
        if paper_tex.exists():
            return paper_tex
        return tex_files[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run a simple sci research workflow using OpenAI models.",
    )
    parser.add_argument("--topic", help="Research topic (auto-detected if existing paper found).")
    parser.add_argument(
        "--output-dir", default="output", help="Directory to store the generated paper and code.",
    )
    parser.add_argument(
        "--model", default="gpt-5", help="OpenAI model to use (default: gpt-5).",
    )
 
    parser.add_argument("--field", help="Research field (auto-detected if existing paper found).")
    parser.add_argument("--question", help="Specific research question to address (auto-detected if existing paper found).")
    parser.add_argument(
        "--modify-existing", 
        action="store_true", 
        help="Force modification of existing paper.tex in output directory, even if it doesn't exist."
    )
    parser.add_argument(
        "--force-new-paper", 
        action="store_true", 
        help="Force creation of new paper even if existing paper.tex is found."
    )
 
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum number of review/evaluation/revision cycles (default: 3)",
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each OpenAI request (default: 3600; use 0 to wait indefinitely)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of times to retry a failed OpenAI request (default: 3)",
    )

    args = parser.parse_args()

    if args.request_timeout == 0:
        args.request_timeout = None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    global client
    client_kwargs = {
        "api_key": api_key,
        "max_retries": args.max_retries,
    }
    if args.request_timeout is not None:
        client_kwargs["timeout"] = args.request_timeout
    client = OpenAI(**client_kwargs)

    try:
        # Check if output directory already contains any .tex file
        output_path = Path(args.output_dir)
        existing_tex_path = find_tex_file(output_path) if output_path.exists() else None
        
        if existing_tex_path and not args.force_new_paper:
            print(f"Found existing paper at {existing_tex_path}")
            print("Starting iterative improvement of existing paper...")
            print("(Use --force-new-paper to create a new paper instead)")
            
            # Read existing paper content
            existing_paper_content = existing_tex_path.read_text(encoding="utf-8")
            
            # Extract metadata from existing paper
            detected_topic, detected_field, detected_question = extract_paper_metadata(existing_paper_content)
            
            # Use detected metadata (no need for user input when existing paper found)
            topic = detected_topic
            field = detected_field  
            question = detected_question
            
            print(f"Detected - Topic: {topic}")
            print(f"Detected - Field: {field}")
            print(f"Detected - Question: {question}")
            
            # Perform iterative improvement without changing core topic/field/question
            paper_content = iteratively_improve_paper(
                existing_paper_content,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            
            # Save the improved paper in the same directory, preserving original filename
            paper_dir = output_path
            paper_path = existing_tex_path
            
            # Create a backup of the original
            backup_name = f"{existing_tex_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
            backup_path = output_path / backup_name
            backup_path.write_text(existing_paper_content, encoding="utf-8")
            print(f"Original paper backed up to {backup_path}")
            
        elif args.modify_existing:
            if not existing_tex_path:
                raise FileNotFoundError(f"--modify-existing specified but no .tex file found in {output_path}")
                
            print(f"Found existing paper at {existing_tex_path}")
            print("Modifying existing paper with new research direction...")
            
            # Ensure we have all required parameters for modification
            if not args.topic:
                args.topic = input("Enter new research topic: ").strip()
            if not args.field:
                args.field = input("Enter research field: ").strip()  
            if not args.question:
                args.question = input("Enter research question: ").strip()
                
            # Read existing paper content
            existing_paper_content = existing_tex_path.read_text(encoding="utf-8")
            
            # Step 1: Generate research idea for modification
            idea = generate_idea(
                args.topic,
                args.field,
                args.question,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            print(f"Generated Idea for Modification:\n{idea}\n")

            # Step 2: Modify existing research paper with new direction
            paper_content = modify_existing_paper(
                existing_paper_content,
                args.topic,
                args.field,
                args.question,
                idea,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            
            # Save the modified paper in the same directory, preserving original filename
            paper_dir = output_path
            paper_path = existing_tex_path
            
            # Create a backup of the original
            backup_name = f"{existing_tex_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
            backup_path = output_path / backup_name
            backup_path.write_text(existing_paper_content, encoding="utf-8")
            print(f"Original paper backed up to {backup_path}")
            
        else:
            print("Creating a new paper...")
            
            # Ensure we have all required parameters
            if not args.topic:
                args.topic = input("Enter research topic: ").strip()
            if not args.field:
                args.field = input("Enter research field: ").strip()
            if not args.question:
                args.question = input("Enter research question: ").strip()
            
            # Step 1: Generate research idea
            idea = generate_idea(
                args.topic,
                args.field,
                args.question,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            print(f"Generated Idea:\n{idea}\n")

            # Step 2: Write research paper
            paper_content = write_paper(
                args.topic,
                args.field,
                args.question,
                idea,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            paper_dir = save_paper_and_code(paper_content, args.output_dir)
            paper_path = paper_dir / "paper.tex"
        
        print(f"Paper saved to {paper_path}")

        # Proceed with review-modify cycles
        for iteration in range(1, args.max_iters + 1):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration}")
            print(f"{'='*50}")
            
            # Review and revise the paper in one step
            feedback, changes, is_ready = review_and_revise_paper(
                paper_content,
                args.model,
                timeout=args.request_timeout,
                max_retries=args.max_retries
            )
            
            print(f"Reviewer Feedback:\n{feedback}\n")
            
            # Check if paper is ready
            if is_ready:
                print("✅ Paper meets publication standards!")
                print(f"Total iterations completed: {iteration}")
                break
            
            # Apply the diff changes
            paper_content = apply_diff_changes(paper_content, changes)
            
            # Save the updated paper with correct filename
            save_paper_and_code(paper_content, paper_dir, paper_path.name)
            print("Paper revised and saved.\n")
        else:
            print("Maximum review iterations reached without editor approval.")
    except KeyboardInterrupt:
        print("Workflow interrupted by user.")
 
 


if __name__ == "__main__":
    main()
