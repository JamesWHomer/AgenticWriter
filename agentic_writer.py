import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

PLANNING_MODEL = "gpt-4.1-nano"
WRITING_MODEL = "gpt-4.1-nano"
SUMMARY_MODEL = "gpt-4.1-nano" 
REASONING_EFFORT = "high"  # low, medium, high

def get_completion(model, system_prompt, user_prompt):
    if model.startswith("o"): # Is a reasoning model
        response = client.responses.create(
            model=model,
            reasoning={"effort": REASONING_EFFORT},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.output_text, {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    else:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7
        }
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content, {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

def get_json_completion(model, system_prompt, user_prompt):
    if model.startswith("o"):
        response = client.responses.create(
            model=model,
            reasoning={"effort": REASONING_EFFORT},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.output_text), {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    else:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7
        }
        
        response = client.chat.completions.create(**params)
        return json.loads(response.choices[0].message.content), {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

def create_book_structure(book_prompt):
    system_prompt = """
You are an expert literary book planner and author. Your task is to craft a sophisticated, award-worthy book blueprint for a FULL-LENGTH NOVEL based on the user's idea.
Return ONLY a valid JSON object with the exact structure below (no additional keys and no prose before/after the JSON):
{
    "title": "Book Title",
    "overview": {
        "book_description": "A high-level synopsis of the entire book, describing tone and stylistic approach",
        "plot_synopsis": "A detailed summary of the complete plot from beginning to end, including all major events and the resolution",
        "themes": ["Theme 1", "Theme 2", ...],
        "literary_elements": ["Element 1", "Element 2", ...],
        "core_characters": [
            {
                "name": "Character Name",
                "role": "Their primary function in the story",
                "arc": "One-sentence summary of their growth or change"
            }
        ],
        "plot_devices": ["Device 1", "Device 2", ...],
        "writing_style": "A brief description of the most appropriate writing style for this book (e.g. clear & concise, lush & poetic, YA, etc.)"
    },
    "chapters": [
        {
            "chapter_number": 1,
            "title": "Chapter Title",
            "description": "An indepth description of what this chapter will accomplish"
        }
    ]
}

IMPORTANT NOVEL LENGTH GUIDELINES:
- Create a full-sized novel structure with 20-30 chapters. Most published novels have at least 20 chapters.
- Ensure the chapters build a complete narrative arc with proper development, climax, and resolution.
- Plan each chapter to be substantial enough for 3,000-5,000 words of content.
- The total novel should be planned for approximately 80,000-100,000 words minimum.

Ensure the plan demonstrates thematic depth, cohesive progression, and is worthy of major literary prizes. The writing_style field should be a short, concrete phrase that best fits the book's concept and audience. Keep the JSON strictly valid.
"""

    user_prompt = f"Create a high-literary full-length novel blueprint for: {book_prompt}"
    result, token_usage = get_json_completion(PLANNING_MODEL, system_prompt, user_prompt)
    print(f"Structure planning: {token_usage['prompt_tokens']} prompt tokens, {token_usage['completion_tokens']} completion tokens, {token_usage['total_tokens']} total tokens used")
    return result, token_usage

def write_chapter(book_structure, chapter_idx, previous_chapters, previous_summaries):
    chapter = book_structure["chapters"][chapter_idx]
    context = ""
    if previous_chapters:
        context += "\n\nPrevious chapter content:\n" + previous_chapters[-1]
    if previous_summaries:
        context += "\n\nSummaries of all previous chapters:\n"
        for i, summary in enumerate(previous_summaries):
            context += f"Chapter {i+1}: {summary}\n"
    style_guidelines = book_structure.get('overview', {}).get('writing_style', 'clear & concise')
    system_prompt = f"""
You are a critically acclaimed novelist known for clear, engaging prose. Write in a style comparable to contemporary literary fiction such as Lauren Groff or Kazuo Ishiguro: vivid yet restrained, prioritizing character and narrative momentum over ornate description. Avoid unnecessary adjectives, mixed metaphors, and 'thesaurus' wording. Clarity and emotional resonance are more important than verbal flourish.

The model-determined writing style for this book is: {style_guidelines}

CHAPTER LENGTH GUIDELINES:
- Write a substantial, full-sized chapter of approximately 3,000-5,000 words.
- This is a full-length novel, not a short story or novella.
- Include appropriate scene breaks if the chapter contains multiple scenes.
- Develop characters, settings, and plot elements thoroughly.
- Ensure the chapter has a clear beginning, middle, and end while advancing the overall narrative.

IMPORTANT OUTPUT INSTRUCTIONS: Output ONLY chapter text without word counts, meta-commentary, or any notes - your output will be automatically integrated into a complete novel.
"""

    user_prompt = f"""
Book Title: {book_structure['title']}

Book Overview:
{json.dumps(book_structure.get('overview', {}), indent=2)}

Full Chapter Outline:
{json.dumps(book_structure['chapters'], indent=2)}

Your task is to write Chapter {chapter['chapter_number']}: {chapter['title']}

Chapter Description: {chapter['description']}
{context}

House Style Guidelines:
  – Use concrete nouns and strong verbs; limit adjectives/adverbs.
  – One figurative image per scene at most.
  – Prefer Anglo-Saxon root words to Latinate where possible.
  – Aim for 6th–9th grade Flesch-Kincaid readability.
  – Avoid purple prose, excess adjectives, repeated imagery, and any wording that seems overwritten.
  – Write a substantial chapter (3,000-5,000 words) as would be found in a published novel.
  – Do NOT include word counts or meta-commentary about the length at the end of your writing.

Please write the complete chapter now, ending it naturally with story content.
"""

    chapter_content, token_usage = get_completion(WRITING_MODEL, system_prompt, user_prompt)
    print(f"Chapter {chapter_idx+1} writing: {token_usage['prompt_tokens']} prompt tokens, {token_usage['completion_tokens']} completion tokens, {token_usage['total_tokens']} total tokens used")
    return chapter_content, token_usage

def summarize_chapter(chapter_content, chapter_info):
    system_prompt = """
    You are an expert at summarizing text. Create a concise summary of the provided chapter, 
    capturing its key points and main narrative elements such that an AI system will be able to most effectively build from. The summary should be no more than 
    150 words.
    """
    
    user_prompt = f"""
    Chapter {chapter_info['chapter_number']}: {chapter_info['title']}
    
    {chapter_content}
    
    Please provide a concise summary of this chapter.
    """
    
    summary, token_usage = get_completion(SUMMARY_MODEL, system_prompt, user_prompt)
    print(f"Chapter {chapter_info['chapter_number']} summary: {token_usage['prompt_tokens']} prompt tokens, {token_usage['completion_tokens']} completion tokens, {token_usage['total_tokens']} total tokens used")
    return summary, token_usage

def sanitize_filename(filename):
    sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
    sanitized = re.sub(r'\s+', "_", sanitized)  
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def save_book_to_file(book_title, chapters):
    filename = f"{sanitize_filename(book_title)}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"# {book_title}\n\n")
            for i, chapter_content in enumerate(chapters):
                file.write(f"## Chapter {i+1}\n\n")
                file.write(chapter_content)
                file.write("\n\n")
        print(f"Successfully saved book to {filename}")
        return filename
    except Exception as e:
        error_filename = "book_output.txt"  
        print(f"Error saving to {filename}: {str(e)}")
        print(f"Attempting to save with alternative filename: {error_filename}")
        with open(error_filename, 'w', encoding='utf-8') as file:
            file.write(f"# {book_title}\n\n")
            for i, chapter_content in enumerate(chapters):
                file.write(f"## Chapter {i+1}\n\n")
                file.write(chapter_content)
                file.write("\n\n")
        return error_filename

def main():
    book_prompt = input("What kind of book would you like to create? ")
    
    if any(model.startswith("o") for model in [PLANNING_MODEL, WRITING_MODEL, SUMMARY_MODEL]):
        global REASONING_EFFORT
        effort_choice = input("\nSelect reasoning effort for 'o' models (low/medium/high, default is medium): ").lower()
        if effort_choice in ["low", "medium", "high"]:
            REASONING_EFFORT = effort_choice
            print(f"Reasoning effort set to: {REASONING_EFFORT}")
        else:
            print(f"Using default reasoning effort: {REASONING_EFFORT}")
    
    print("\nPlanning your book structure...")
    book_structure, structure_tokens = create_book_structure(book_prompt)
    print(f"\n=== BOOK STRUCTURE ===\n")
    print(json.dumps(book_structure, indent=2, ensure_ascii=False))
    print("\n=== BOOK OVERVIEW ===\n")
    overview = book_structure.get('overview', {})
    print(f"Title: {book_structure.get('title', '')}")
    print(f"Description: {overview.get('book_description', '')}")
    print(f"Plot Synopsis: {overview.get('plot_synopsis', '')}")
    print(f"Themes: {overview.get('themes', [])}")
    print(f"Literary Elements: {overview.get('literary_elements', [])}")
    print(f"Core Characters: {overview.get('core_characters', [])}")
    print(f"Plot Devices: {overview.get('plot_devices', [])}")
    print(f"\033[1mWriting Style: {overview.get('writing_style', '').upper()}\033[0m\n")
    print(f"This book will have {len(book_structure['chapters'])} chapters:\n")
    for chapter in book_structure['chapters']:
        print(f"Chapter {chapter['chapter_number']}: {chapter['title']}")
        print(f"  {chapter['description'][:100]}...\n")
    proceed = input("\nReady to write this book? (yes/no): ")
    if proceed.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    chapters_content = []
    chapter_summaries = []
    chapter_tokens = []
    summary_tokens = []
    
    total_prompt_tokens = structure_tokens["prompt_tokens"]
    total_completion_tokens = structure_tokens["completion_tokens"]
    total_tokens = structure_tokens["total_tokens"]
    
    for i, chapter in enumerate(book_structure['chapters']):
        print(f"\nWriting Chapter {i+1}: {chapter['title']}...")
        chapter_content, chapter_token_usage = write_chapter(
            book_structure, 
            i, 
            chapters_content,
            chapter_summaries
        )
        chapters_content.append(chapter_content)
        chapter_tokens.append(chapter_token_usage)
        
        print(f"Summarizing Chapter {i+1}...")
        summary, summary_token_usage = summarize_chapter(chapter_content, chapter)
        chapter_summaries.append(summary)
        summary_tokens.append(summary_token_usage)
        
        total_prompt_tokens += chapter_token_usage["prompt_tokens"] + summary_token_usage["prompt_tokens"]
        total_completion_tokens += chapter_token_usage["completion_tokens"] + summary_token_usage["completion_tokens"]
        total_tokens += chapter_token_usage["total_tokens"] + summary_token_usage["total_tokens"]
        
        progress_file = save_book_to_file(book_structure['title'], chapters_content)
        print(f"Progress saved to {progress_file}")
        print(f"Cumulative token usage: {total_prompt_tokens} prompt, {total_completion_tokens} completion, {total_tokens} total")
    
    print(f"\nBook '{book_structure['title']}' completed!")
    print(f"Final file saved as: {progress_file}")
    print(f"\nTotal tokens used: {total_prompt_tokens} prompt, {total_completion_tokens} completion, {total_tokens} total")
    
    metadata_filename = f"{sanitize_filename(book_structure['title'])}_metadata.json"
    try:
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "title": book_structure['title'],
                "chapters": book_structure['chapters'],
                "summaries": chapter_summaries,
                "token_usage": {
                    "structure": structure_tokens,
                    "chapters": chapter_tokens,
                    "summaries": summary_tokens,
                    "total": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
            }, f, indent=2)
        print(f"Metadata saved to {metadata_filename}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")
        fallback_metadata = "book_metadata.json"
        with open(fallback_metadata, 'w', encoding='utf-8') as f:
            json.dump({
                "title": book_structure['title'],
                "chapters": book_structure['chapters'],
                "summaries": chapter_summaries,
                "token_usage": {
                    "structure": structure_tokens,
                    "chapters": chapter_tokens,
                    "summaries": summary_tokens,
                    "total": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
            }, f, indent=2)
        print(f"Metadata saved to fallback file: {fallback_metadata}")

if __name__ == "__main__":
    main() 