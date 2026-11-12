"""
benchmarks/english_dataset.py
TinyStories-style connected English narrative dataset generator & splitter.
Produces a structured, coherent multi-sentence English corpus of 25,000+ tokens
across hundreds of narrative paragraphs with:
  - Train Split (90%): Rich narrative stories covering recurring characters, settings,
    actions, causal chains, and moral conclusions.
  - Validation Split (10%): Held-out narrative stories composed of unseen compositional
    combinations from the same grammar/vocabulary to measure true generalization
    (Validation Perplexity & Next-Token Accuracy).
Strictly zero autograd.
"""
import random


# Compositional narrative building blocks (TinyStories style)
OPENINGS = [
    "Once upon a time in the green valley",
    "In the bright morning near the quiet river",
    "One sunny afternoon in the small village",
    "Under the tall oak tree in the ancient forest",
    "Early at dawn beside the blue lake",
    "Near the golden meadow on a gentle hill",
    "Inside the cozy cottage by the warm garden",
    "High on the misty mountain above the clouds",
]

CHARACTERS = [
    ("the little bird", "sang a sweet song", "flew across the clear sky", "built a safe nest in the branches"),
    ("the young rabbit", "hopped along the soft path", "found a fresh carrot in the grass", "shared the food with his friends"),
    ("the clever fox", "walked quietly through the woods", "watched the silver fish in the stream", "learned a wise lesson about patience"),
    ("the brave puppy", "ran happily around the yard", "chased a colorful butterfly near the flowers", "guarded the wooden gate all day"),
    ("the gentle deer", "drank cool water from the brook", "rested beneath the shady willow tree", "guided the lost fawn back home"),
    ("the busy squirrel", "gathered shiny acorns for the winter", "climbed up the tallest pine tree", "stored the nuts inside a hollow log"),
    ("the kind girl", "planted bright yellow seeds in the soil", "watered the blooming roses every morning", "smiled when the garden grew tall"),
    ("the curious boy", "built a small wooden boat by the shore", "sailed the toy boat across the calm pond", "Cheered as the wind pushed the white sail"),
    ("the wise owl", "sat high on the old stone tower", "listened to the quiet night wind", "showed the travelers the safe road"),
    ("the playful kitten", "chased a rolling red ball on the floor", "curled up on the soft warm blanket", "purred softly beside the glowing fireplace"),
]

MIDDLES = [
    "Soon the gentle wind blew softly and brought the scent of spring flowers.",
    "Along the way a friendly neighbor waved and offered a basket of ripe apples.",
    "Suddenly a soft rain began to fall and made the green leaves shine brightly.",
    "After a short rest the sun came out again and warmed the whole valley.",
    "Every creature in the forest gathered together to celebrate the peaceful day.",
    "With careful steps and a happy heart the journey continued toward the bridge.",
]

CLOSINGS = [
    "In the end everyone felt grateful and returned home before the stars appeared.",
    "When evening came the sky turned orange and peace filled the quiet village.",
    "That night the moon shone brightly over the hills while everyone slept safely.",
    "From that day on the friends helped each other and lived in harmony.",
    "Finally the work was finished and joy echoed across the sunny meadow.",
]

# Canonical Seed Prompts used for Free-Running Generation Audit after Production Freeze
CANONICAL_SEED_PROMPTS = [
    "Once upon a time",
    "The little bird",
    "In the bright morning",
    "The young rabbit",
    "Under the tall oak",
]


def build_english_tinystories_corpus(
    target_min_tokens: int = 24000,
    train_ratio: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Generates a structured TinyStories-style English corpus of >= `target_min_tokens`
    tokens, partitioned into 90% Train and 10% Validation (unseen combinations of
    openings, characters, middles, and closings).
    """
    rng = random.Random(seed)

    # Generate all possible structured story combinations
    all_combinations = []
    for o_idx, opening in enumerate(OPENINGS):
        for c_idx, (subj, act1, act2, act3) in enumerate(CHARACTERS):
            for m_idx, middle in enumerate(MIDDLES):
                cl_idx = (o_idx + c_idx + m_idx) % len(CLOSINGS)
                closing = CLOSINGS[cl_idx]

                # Construct two syntactic variants (Subject-first & Opening-first)
                if (o_idx + m_idx) % 2 == 0:
                    story = (
                        f"{opening} , {subj} {act1} . "
                        f"Then {subj} {act2} and {act3} . "
                        f"{middle} {closing}"
                    )
                else:
                    story = (
                        f"{subj.capitalize()} {act1} {opening.lower()} . "
                        f"Afterward {subj} {act2} and {act3} . "
                        f"{middle} {closing}"
                    )
                all_combinations.append((o_idx, c_idx, m_idx, story))

    rng.shuffle(all_combinations)

    # Hold out 10% unseen combinations for Validation, use 90% for Train
    n_Unique = len(all_combinations)
    n_train_unique = int(n_Unique * train_ratio)

    train_unique = [item[3] for item in all_combinations[:n_train_unique]]
    val_unique = [item[3] for item in all_combinations[n_train_unique:]]

    # Repeat/sample stories to reach target token scale (e.g. 24,000+ tokens)
    train_stories: list[str] = []
    val_stories: list[str] = []

    train_word_count = 0
    target_train_words = int(target_min_tokens * train_ratio)
    idx = 0
    while train_word_count < target_train_words:
        s = train_unique[idx % len(train_unique)]
        train_stories.append(s)
        train_word_count += len(s.split())
        idx += 1

    val_word_count = 0
    target_val_words = int(target_min_tokens * (1.0 - train_ratio))
    idx = 0
    while val_word_count < target_val_words:
        s = val_unique[idx % len(val_unique)]
        val_stories.append(s)
        val_word_count += len(s.split())
        idx += 1

    return {
        "train_stories": train_stories,
        "val_stories": val_stories,
        "unique_train_stories": train_unique,
        "unique_val_stories": val_unique,
        "train_word_count": train_word_count,
        "val_word_count": val_word_count,
        "total_word_count": train_word_count + val_word_count,
        "seed_prompts": CANONICAL_SEED_PROMPTS,
    }
