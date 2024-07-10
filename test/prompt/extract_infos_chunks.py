import sys
import asyncio

sys.path.append("/Users/remillieux/Documents/TowardsChange/FRIDAYGPT/FridayGPT/FA_LLM/")

from modules.llm import llm_extract_info_chunk, create_chat_prompt_new
from modules.prompt import extract_info_chunk_instructions

question = "C'est quoi le prénom de VINEL ?"

context = """Name+H23A1:J30: Nathalie MARCHAL-GEORGE; Nom: MARCHAL-GEORGE; Prénom: Nathalie; BD: POP; Poste: Principal Advisor; Expertise domain: Start-up; Concept: ; Niveau: 1; Technologies: 
Name+H23A1:J30: Daniel-Jean VINEL; Nom: VINEL; Prénom: Daniel-Jean; BD: T&TS; Poste: Principal Technologist; Expertise domain: Olefins; Concept: ; Niveau: 1; Technologies: Dimersol™-E, Dimersol™-G, Dimersol™- X, AlphaButol®, AlphaHexol™
Name+H23A1:J30: Etienne NIDERKORN; Nom: NIDERKORN; Prénom: Etienne; BD: T&TS; Poste: Principal Technology Engineer; Expertise domain: Olefins; Concept: ; Niveau: 1; Technologies: 
Name+H23A1:J30: Boris HESSE; Nom: HESSE; Prénom: Boris; BD: T&TS; Poste: Clean Fuels Technology Engineer; Expertise domain: Clean Fuels; Concept: ; Niveau: 1; Technologies:"""

message = extract_info_chunk_instructions.format(question=question, source=context)


async def run():
    prompt = create_chat_prompt_new(
        system_prompt=llm_extract_info_chunk.system_prompt,
        memory_list=None,
        context=None,
        instruction=None,
        message=message,
    )

    result = await llm_extract_info_chunk.inference(prompt=prompt)

    print(result)


asyncio.run(run())
