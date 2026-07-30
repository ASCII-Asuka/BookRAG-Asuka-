from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ParsedReactResponse:
    thought: str
    action: str
    complete: bool


# ReAct ICLR 2023 official six-shot HotpotQA prompt, MIT licensed.
# Source commit: 6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9
HOTPOTQA_REACT_EXAMPLES = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr. It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action 3: Finish[yes]
""".strip()


QASPER_DEMO_IDS = (
    "753990d0b621d390ed58f20c4d9e4f065f0dc672",
    "44c4bd6decc86f1091b5fc0728873d9324cdde4e",
    "003f884d3893532f8c302431c9f70be6f64d9be8",
    "938cf30c4f1d14fa182e82919e16072fdbcf2a82",
    "19c9cfbc4f29104200393e848b7b9be41913a7ac",
    "cd1034c183edf630018f47ff70b48d74d2bb1649",
)


QASPER_REACT_EXAMPLES = """
Question: What is the seed lexicon?
Thought 1: I should inspect the section that defines discourse relation-based event pairs.
Action 1: Search[Proposed Method ::: Discourse Relation-Based Event Pairs]
Observation 1: Our method requires a very small seed lexicon and a large raw corpus. We assume that we can automatically extract discourse-tagged event pairs from the raw corpus.
Thought 2: I should locate the exact definition of the seed lexicon.
Action 2: Lookup[seed lexicon]
Observation 2: (Result 1 / 2) Our method requires a very small seed lexicon and a large raw corpus.
Thought 3: I need the next occurrence containing the definition.
Action 3: Lookup[seed lexicon]
Observation 3: (Result 2 / 2) The seed lexicon consists of positive and negative predicates.
Thought 4: The definition is explicit.
Action 4: Finish[a vocabulary of positive and negative predicates]
Question: How big is the Japanese data?
Thought 1: I should inspect the dataset statistics section.
Action 1: Search[Experiments ::: Dataset ::: AL, CA, and CO]
Observation 1: From the corpus of about 100 million sentences, we obtained 1.4 millions event pairs for AL, 41 millions for CA, and 6 millions for CO. We sampled the CA and CO event pairs for the reported dataset.
Thought 2: I should locate the Japanese Web corpus description and reported data size.
Action 2: Lookup[corpus]
Observation 2: (Result 1 / 1) From the corpus of about 100 million sentences, we obtained 1.4 millions event pairs for AL, 41 millions for CA, and 6 millions for CO.
Thought 3: The reported sampled Japanese data contains about seven million event pairs.
Action 3: Finish[7,000,000 event pairs]
Question: Do they report results only on English data?
Thought 1: I should inspect how the Reddit dataset handles language.
Action 1: Search[Applying the typology to Reddit]
Observation 1: Our full dataset consists of all subreddits on Reddit from January 2013 to December 2014, and we manually remove communities where the bulk of the contributions are in a foreign language.
Thought 2: I should confirm the language filtering statement.
Action 2: Lookup[foreign language]
Observation 2: (Result 1 / 1) We manually remove communities where the bulk of the contributions are in a foreign language.
Thought 3: The dataset is not described as reporting only on English data.
Action 3: Finish[No]
Question: How do the authors measure how temporally dynamic a community is?
Thought 1: I should inspect the community-level measures section.
Action 1: Search[Community-level measures]
Observation 1: Dynamicity. A highly dynamic community constantly shifts interests from one time window to another, and these temporal variations are reflected in its use of volatile language.
Thought 2: I should find the formal definition of dynamicity.
Action 2: Lookup[temporally dynamic]
Observation 2: No more results.
Thought 3: I should instead look up volatility.
Action 3: Lookup[volatility]
Observation 3: (Result 1 / 1) Formally, we define the dynamicity of a community as the average volatility of all utterances.
Thought 4: The definition directly answers the question.
Action 4: Finish[the average volatility of all utterances]
Question: How many questions are in the dataset?
Thought 1: I should inspect the dataset and evaluation metrics section.
Action 1: Search[Experimental Studies ::: Dataset and Evaluation Metrics]
Observation 1: Our dataset is annotated based on Chinese pathology reports provided by the Department of Gastrointestinal Surgery, Ruijin Hospital.
Thought 2: I should find the number of question-answer pairs.
Action 2: Lookup[question-answer pairs]
Observation 2: (Result 1 / 1) It contains 17,833 sentences, 826,987 characters and 2,714 question-answer pairs.
Thought 3: The dataset therefore has 2,714 questions.
Action 3: Finish[2,714]
Question: Does their detection tool work better than human detection?
Thought 1: I should inspect the generative model section containing the human and machine detection comparison.
Action 1: Search[Generative Model]
Observation 1: The fake reviews were detected by human participants with a class-averaged F-score of only 56 percent, which is close to random detection.
Thought 2: I should locate the machine detector result.
Action 2: Lookup[97%]
Observation 2: (Result 1 / 1) The fake reviews that users had most difficulty detecting are detected with an excellent 97 percent F-score.
Thought 3: The automatic detector performs better than the human participants.
Action 3: Finish[Yes]
""".strip()


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
(1) Search[query], which searches the local document collection and returns the first evidence units of the best matching page.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current page.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""


def parse_react_response(raw, step):
    text = str(raw or "").strip()
    thought_match = re.search(
        rf"(?:^|\n)Thought\s*{step}\s*:\s*"
        rf"(.*?)(?=\nAction\s*{step}\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        rf"(?:^|\n)Action\s*{step}\s*:\s*"
        rf"((?:Search|Lookup|Finish)\[.*\])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    thought = (
        thought_match.group(1).strip()
        if thought_match
        else (text.splitlines()[0].strip() if text else "")
    )
    action = action_match.group(1).strip() if action_match else ""
    return ParsedReactResponse(
        thought=thought,
        action=action,
        complete=bool(action),
    )


def get_react_examples(dataset_name, prompt_file=""):
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    name = str(dataset_name or "").lower()
    if "hotpot" in name:
        return HOTPOTQA_REACT_EXAMPLES
    if "qasper" in name:
        return QASPER_REACT_EXAMPLES
    raise ValueError(f"Unsupported ReAct dataset: {dataset_name}")


def build_react_prompt(
    dataset_name,
    question,
    trajectory,
    step,
    prompt_file="",
):
    prefix = (
        REACT_INSTRUCTION
        + get_react_examples(
            dataset_name,
            prompt_file=prompt_file,
        ).rstrip()
        + "\n"
    )
    current = f"Question: {str(question).strip()}\n{trajectory}"
    return prefix + current + f"Thought {step}:"
