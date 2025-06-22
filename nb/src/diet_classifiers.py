import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import re
import pandas as pd
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Pipeline
from typing import Set
from typing import List, Dict, Any, Set, Optional
from thefuzz import process, fuzz
from typing import Dict, Any, Optional, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
from sklearn.metrics import classification_report,confusion_matrix

        
# ==============================================================================
#  Vegan classifier
# ==============================================================================

UNITS: Set[str] = {
    "c", "cup", "cups",
    "g", "gram", "grams",
    "kg", "kilogram", "kilograms",
    "l", "liter", "liters",
    "lb", "lbs", "pound", "pounds",
    "ml", "milliliter", "milliliters",
    "oz", "ounce", "ounces",
    "pinch", "pinches",
    "splash", "splashes",
    "sprig", "sprigs",
    "t", "tsp", "teaspoon", "teaspoons",
    "T", "tbsp", "tablespoon", "tablespoons",
    "can", "cans",
    "clove", "cloves",
    "dash", "dashes",
    "drizzle",
    "drop", "drops",
    "gallon", "gallons",
    "handful", "handfuls",
    "head", "heads",
    "package", "packages",
    "packet", "packets",
    "pint", "pints",
    "quart", "quarts",
    "scoop", "scoops",
    "sheet", "sheets",
    "slice", "slices",
    "stalk", "stalks",
    "stick", "sticks",
    "strip", "strips",
}
# A comprehensive and categorized set of non-essential words found in ingredient lists.
# The purpose of this set is to remove these words from an ingredient string
# to help isolate the core, identifiable name of the food item.
DESCRIPTORS: Set[str] = {
    # --- Preparation & Actions ---
    'beaten', 'blanched', 'boiled', 'braised', 'brewed', 'brined', 'broken',
    'charred', 'chilled', 'chopped', 'clarified', 'coarsely', 'crumbled', 'crushed',
    'cubed', 'cut', 'deboned', 'deglazed', 'deseeded', 'deveined', 'diced',
    'dissolved', 'divided', 'drained', 'finely', 'flaked', 'folded', 'grated',
    'grilled', 'ground', 'halved', 'heated', 'hulled', 'husked', 'infused',
    'julienned', 'juiced', 'kneaded', 'marinated', 'mashed', 'melted', 'minced',
    'mixed', 'parboiled', 'patted', 'peeled', 'pitted', 'poached', 'pounded',
    'prepared', 'pressed', 'pureed', 'quartered', 'rinsed', 'roasted', 'rolled',
    'roughly', 'scalded', 'scored', 'scrubbed', 'seared', 'seeded', 'segmented',
    'shaved', 'shredded', 'shucked', 'sifted', 'skewered', 'sliced', 'slivered',
    'smashed', 'soaked', 'softened', 'squeezed', 'steamed', 'stemmed', 'stewed',
    'strained', 'stuffed', 'thawed', 'thinly', 'tied', 'toasted', 'torn', 'trimmed',
    'whisked', 'zested',

    # --- State, Condition & Temperature ---
    'canned', 'cold', 'condensed', 'cooked', 'cooled', 'cored', 'creamed', 'cured',
    'defrosted', 'dried', 'fermented', 'firmly', 'fresh', 'freshly', 'frozen',
    'hard', 'hot', 'instant', 'jarred', 'lean', 'leftover', 'light', 'lukewarm',
    'optional', 'pasteurized', 'powdered', 'preserved', 'raw', 'ready-to-use',
    'refrigerated', 'ripe', 'room', 'skin-on', 'skinless', 'soft', 'stiff',
    'temperature', 'uncooked', 'undrained', 'unripe', 'warm', 'washed', 'whole',

    # --- Size & Shape ---
    'bite-sized', 'chunky', 'clump', 'coarse', 'fine', 'jumbo', 'large', 'long',
    'medium', 'round', 'short', 'small', 'thick', 'thin',

    # --- Quantifiers & Qualifiers ---
    'about', 'additional', 'approximately', 'bunch', 'coarse', 'extra', 'generous',
    'heavy', 'heaping', 'level', 'more', 'packed', 'plus', 'scant', 'splash',
    'sprig', 'sprinkle',

    # --- Flavor & Taste ---
    'bitter', 'salty', 'savory', 'sour', 'spicy', 'sweet', 'sweetened', 'unsalted',
    'unsweetened',

    # --- Common Stop Words (Articles, Conjunctions, Prepositions) ---
    'a', 'an', 'and', 'as', 'at', 'for', 'in', 'into', 'of', 'on', 'or', 'the',
    'to', 'with', 'without',

    # --- Instructions & Meta-words ---
    'divided', 'dusting', 'garnish', 'needed', 'serving', 'taste',
}

def parse_ingredient(ingredient_string: str) -> str:
    """
    Parses a raw ingredient string to extract its essential name.

    This function cleans the input string by performing a series of sequential
    operations:
    1.  Converts the string to lowercase.
    2.  Removes text within parentheses (e.g., "(optional)").
    3.  Removes numerical quantities, including fractions and decimals.
    4.  Removes punctuation.
    5.  Splits the string into words and removes common units and descriptors.
    6.  Reassembles the string and normalizes whitespace.

    Args:
        ingredient_string: The raw ingredient string from a recipe.
                           Example: "2 1/2 cups (12.5 oz) sifted all-purpose flour, for dusting"

    Returns:
        A cleaned, normalized string representing the core ingredient.
        Example: "all-purpose flour"
    """
    if not isinstance(ingredient_string, str) or not ingredient_string:
        return ""

    # 1. Convert to lowercase for consistent processing.
    text = ingredient_string.lower()

    # 2. Remove parenthetical remarks (e.g., "(optional)", "(about 1 pound)").
    text = re.sub(r'\([^)]*\)', '', text)

    # 3. Remove numerical quantities, including integers, decimals, and fractions.
    # This regex handles formats like "1 1/2", "1/2", "1.5", "1".
    text = re.sub(r'(\d+\s+)?\d+/\d+|\d+(\.\d+)?|\d+', '', text)

    # 4. Remove common punctuation. We keep hyphens as they can be part of a name.
    text = re.sub(r'[,.;:?!"]', '', text)

    # 5. Tokenize and filter out units and descriptors.
    words = text.split()
    # This list comprehension is efficient for filtering. We check against the
    # predefined sets of UNITS and DESCRIPTORS.
    clean_words = [
        word for word in words if word not in UNITS and word not in DESCRIPTORS
    ]

    # 6. Reassemble the string and clean up whitespace.
    # ' '.join() handles the list-to-string conversion.
    # The final split/join is a robust way to normalize multiple spaces to single spaces.
    clean_name = ' '.join(clean_words).strip()

    return clean_name



# ==============================================================================
#  Global State and Constants for Classifier
# ==============================================================================

# Caching mechanism to store results and avoid re-computation, critical for performance.
# Key: clean ingredient name (str), Value: vegan status (bool)
VEGAN_CACHE: Dict[str, bool] = {}

# Lazy-loaded Hugging Face pipeline. Initialized only when first needed.
CLASSIFIER_PIPELINE: Optional[Pipeline] = None

# --- Rule-Based Keyword Sets ---

# Keywords for ingredients that are definitively NOT vegan.
# This list is comprehensive to catch common animal products quickly.
NON_VEGAN_KEYWORDS: Set[str] = {
    # --- Meats (Red & White) ---
    'andouille', 'bacon', 'beef', 'biltong', 'bison', 'boar', 'bologna', 'bratwurst', 'brisket', 'capicola', 'chorizo', 'chops', 'corned beef', 'frankfurter', 'goat', 'ground chuck', 'guanciale', 'ham', 'head cheese', 'jerky', 'kebab', 'kielbasa', 'kidney', 'lamb', 'liver', 'meat', 'meatball', 'meatballs', 'mince', 'mortadella', 'mutton', 'pancetta', 'pastrami', 'pemmican', 'pepperoni', 'pork', 'prosciutto', 'ribs', 'salami', 'sausage', 'shank', 'soppressata', 'steak', 'sweetbreads', 'tenderloin', 'tongue', 'tripe', 'veal', 'venison',
    # --- Poultry ---
    'albumen', 'capon', 'chicken', 'confit', 'cornish hen', 'duck', 'egg', 'eggs', 'foie gras', 'giblets', 'goose', 'guinea fowl', 'meringue', 'nuggets', 'ostrich', 'partridge', 'pate', 'pheasant', 'poultry', 'quail', 'turkey', 'yolk',
    # --- Seafood (Fish) ---
    'anchovies', 'bass', 'bluefish', 'carp', 'catfish', 'caviar', 'cod', 'eel', 'escargot', 'fish', 'flounder', 'gefilte fish', 'grouper', 'haddock', 'halibut', 'herring', 'lox', 'mahi-mahi', 'mackerel', 'monkfish', 'perch', 'pickerel', 'pollock', 'roe', 'salmon', 'sardine', 'seabass', 'sole', 'sturgeon', 'surimi', 'swordfish', 'tilapia', 'trout', 'tuna', 'walleye',
    # --- Seafood (Shellfish & Other) ---
    'abalone', 'calamari', 'clams', 'cockle', 'conch', 'crab', 'crawfish', 'crayfish', 'cuttlefish', 'krill', 'langostino', 'lobster', 'mussels', 'octopus', 'oyster', 'oysters', 'prawns', 'scallop', 'scallops', 'scampi', 'sea urchin', 'seafood', 'shrimp', 'squid', 'uni', 'whelk',
    # --- Dairy ---
    'asiago', 'bleu', 'brie', 'goat cheese', 'blue cheese', 'butter', 'buttermilk', 'camembert', 'casein', 'caseinate', 'cheddar', 'cheese', 'colby', 'cottage', 'cream', 'creme', 'curd', 'edam', 'feta', 'ghee', 'gorgonzola', 'gouda', 'gruyere', 'half-and-half', 'halloumi', 'havarti', 'kefir', 'lactalbumin', 'lactose', 'manchego', 'mascarpone', 'milk', 'monterey jack', 'mozzarella', 'muenster', 'neufchatel', 'paneer', 'parmesan', 'provolone', 'queso', 'ricotta', 'sour cream', 'whey', 'yogurt',
    # --- Animal Fats, By-products & Additives ---
    'ambergris', 'aspic', 'bone char', 'bone meal', 'bone marrow', 'bouillon', 'broth', 'carmine', 'chitin', 'cochineal', 'collagen', 'consomme', 'demi-glace', 'drippings', 'fat', 'fish oil', 'gelatin', 'glycerides', 'glycerol', 'isinglass', 'keratin', 'l-cysteine', 'lanolin', 'lard', 'lipase', 'musk', 'pepsin', 'rennet', 'schmaltz', 'shellac', 'stearic acid', 'stock', 'suet', 'tallow', 'vitamin d3',
    # --- Bee Products ---
    'bee pollen', 'beeswax', 'honey', 'propolis', 'royal jelly',
}

# --- ALWAYS_VEGAN_KEYWORDS ---
# A comprehensive list of common ingredients that are almost always vegan. 
# This helps quickly classify simple ingredients without needing the ML model.
ALWAYS_VEGAN_KEYWORDS: Set[str] = {
    # --- Staples & Dry Goods ---
    'arrowroot', 'baking powder', 'baking soda', 'beans', 'bread crumbs', 'chickpeas', 'cornmeal', 'cornstarch', 'couscous', 'flour', 'lentils', 'pasta', 'quinoa', 'rice', 'sugar', 'yeast',
    # --- Fats & Oils ---
    'margarine', 'oil', 'canola oil', 'coconut oil', 'olive oil', 'sesame oil', 'sunflower oil', 'vegetable oil',
    # --- Seasonings, Herbs & Spices ---
    'basil', 'black pepper', 'cayenne', 'chili', 'cilantro', 'cinnamon', 'clove', 'cumin', 'curry', 'garlic', 'ginger', 'herbs', 'mustard', 'nutmeg', 'onion', 'oregano', 'paprika', 'parsley', 'pepper', 'rosemary', 'saffron', 'salt', 'spices', 'thyme', 'turmeric',
    # --- Liquids & Acids ---
    'coffee', 'club soda', 'lemon juice', 'lime juice', 'seltzer', 'soda', 'tea', 'vegetable stock', 'vegetable broth', 'vinegar', 'water','rice milk','vegan margarine'
}


# --- VEGAN_ALTERNATIVE_PREFIXES ---
# A list of prefixes for plant-based alternatives that can otherwise be misidentified
# by simple keyword matching (e.g., 'soy milk', 'cashew cheese').
VEGAN_ALTERNATIVE_PREFIXES: Set[str] = {
    # --- Nuts & Seeds ---
    'almond', 'cashew', 'flax', 'hazelnut', 'hemp', 'macadamia', 'pecan', 'pistachio', 'pumpkinseed', 'sesame', 'sunflower', 'walnut',
    # --- Grains & Legumes ---
    'chickpea', 'lentil', 'oat', 'pea', 'quinoa', 'rice', 'soy',
    # --- Fruits & Vegetables ---
    'apple', 'avocado', 'banana', 'potato', 'vegetable', 'veggie',
    # --- Other ---
    'cocoa', 'coconut', 'plant', 'plant-based', 'vegan','non-dairy','vegan margarine', 'rice milk'
}

ALWAYS_KETO_KEYWORDS = {
    'splenda', 'stevia', 'erythritol', 'monkfruit', 'sucralose', 'allulose'
}
def _get_classifier() -> Optional[Pipeline]:
    """
    Initializes and returns the Hugging Face text classification pipeline.

    This function uses lazy loading: the model is only loaded into memory
    the first time it's needed, preventing slow startup times.
    It is robust against the absence of the `transformers` library.
    """
    global CLASSIFIER_PIPELINE
    if CLASSIFIER_PIPELINE is None:
        if pipeline is None:
            # transformers library is not installed
            return None
        try:
            print("INFO: Initializing Hugging Face classifier for the first time. This may take a moment...")
            # Using the model mentioned in the task prompt.
            CLASSIFIER_PIPELINE  = pipeline("text-classification", model="nisuga/food_type_classification_model")
            print("INFO: Classifier initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load Hugging Face model nisuga/food_type_classification_model. ML classification will be disabled.")
            print(f"Error details: {e}")
            # Set to a dummy value to prevent re-trying
            CLASSIFIER_PIPELINE = "failed"
    
    if CLASSIFIER_PIPELINE == "failed":
        return None
        
    return CLASSIFIER_PIPELINE


def is_ingredient_vegan(ingredient_string: str) -> bool:
    """
    Classifies a single ingredient as vegan or not using a multi-layered approach.

    The process is as follows:
    1. Parse the ingredient to get a clean name.
    2. Check a cache for a previously computed result.
    3. Apply a series of precise rules to handle common cases and pitfalls
       (e.g., "eggless", "peanut butter", "soy milk").
    4. If no rule applies, use a Hugging Face ML model for classification.
    5. Cache the final result before returning.

    Args:
        ingredient_string: The raw ingredient string from a recipe.

    Returns:
        True if the ingredient is determined to be vegan, False otherwise.
    """

    # 1. Parse the ingredient string to get a clean, standardized name.
    clean_name = parse_ingredient(ingredient_string)

    if not clean_name:
        return True  # An empty or unparsable ingredient is assumed to not make a dish non-vegan.
    
    # goat cheese
    if "goat cheese" in clean_name:
        return False

    # 2. Check cache for a quick result.
    if clean_name in VEGAN_CACHE:
        return VEGAN_CACHE[clean_name]

    # --- 3. Apply Rule-Based Logic (Fast and Accurate Checks) ---
    
    # Pitfall: Handle "eggless" before checking for "egg".
    if "eggless" in clean_name:
        VEGAN_CACHE[clean_name] = True
        return True

    # Pitfall: Handle plant-based butters/milks before general "butter"/"milk" checks.
    if any(prefix in clean_name for prefix in VEGAN_ALTERNATIVE_PREFIXES):
        # e.g., "peanut butter", "soy milk"
        VEGAN_CACHE[clean_name] = True
        
        return True

    # General check for common non-vegan keywords.
    # We split the clean name to avoid false positives (e.g., "ham" in "shame").
    words_in_name = set(clean_name.split())
    if not NON_VEGAN_KEYWORDS.isdisjoint(words_in_name):
        VEGAN_CACHE[clean_name] = False
        return False

    # Check for always-vegan ingredients.
    if clean_name in ALWAYS_VEGAN_KEYWORDS:
        VEGAN_CACHE[clean_name] = True
        return True

    # --- 4. Fallback to ML Model for Ambiguous Cases ---
    # Assume non-vegan as a safe default if ML model fails or is unavailable.
    result = False
    
    classifier = _get_classifier()

    if classifier:
        try:
            prediction = classifier(clean_name, top_k=1)[0]
            # The model labels are 'plant-based' and 'animal-based'.
            if prediction['label'] == 'PLANT_BASED':
                result = True
        except Exception as e:
            print(f"WARNING: ML classification failed for '{clean_name}'. Defaulting to non-vegan. Error: {e}")
            result = False
    else:
        print(f"WARNING: No ML classifier available. Defaulting '{clean_name}' to non-vegan.")
        result = False

    # 5. Cache the result before returning.
    VEGAN_CACHE[clean_name] = result
    return result


# ==============================================================================
# SECTION 2: KETO CLASSIFIER
# ==============================================================================

# --- Tuning Parameters & Knowledge Bases ---
LOG_FAILED_LOOKUPS = False


# FIXED: Updated keyword lists with better coverage
MANUAL_KETO_OVERRIDES = {
    # Beverages
    'water', 'sparkling water', 'black coffee', 'tea', 'green tea', 'herbal tea',
    'bone broth', 'chicken broth', 'beef broth', 'vegetable broth',
    
    # Basic ingredients & seasonings
    'salt', 'black pepper', 'white pepper', 'sea salt', 'himalayan salt',
    'garlic', 'garlic powder', 'onion powder', 'paprika', 'cumin', 'oregano',
    'basil', 'thyme', 'rosemary', 'sage', 'cilantro', 'parsley', 'dill',
    'cinnamon', 'nutmeg', 'ginger', 'turmeric', 'cayenne', 'chili powder',
    'herbs', 'spice', 'vanilla extract', 'lemon extract',
    
    # Fats & oils
    'oil', 'olive oil', 'coconut oil', 'avocado oil', 'mct oil', 'sesame oil',
    'butter', 'ghee', 'lard', 'tallow', 'duck fat', 'macadamia oil',
    
    # Proteins - Meat
    'chicken', 'chicken breast', 'chicken thigh', 'chicken wings', 'turkey',
    'beef', 'ground beef', 'steak', 'ribeye', 'sirloin', 'brisket',
    'pork', 'pork chops', 'pork belly', 'bacon', 'ham', 'prosciutto',
    'lamb', 'lamb chops', 'ground lamb', 'venison', 'duck', 'goose',
    'pepperoni', 'salami', 'chorizo', 'sausage', 'bratwurst',
    
    # Proteins - Seafood
    'fish', 'salmon', 'tuna', 'cod', 'halibut', 'mahi mahi', 'sea bass',
    'trout', 'mackerel', 'sardines', 'anchovies', 'herring',
    'shrimp', 'crab', 'lobster', 'scallops', 'mussels', 'oysters',
    'calamari', 'octopus', 'clams',
    
    # Proteins - Other
    'egg', 'eggs', 'egg white', 'egg yolk', 'quail eggs',
    
    # Dairy & cheese
    'cheese', 'cheddar cheese', 'parmesan cheese', 'mozzarella cheese',
    'swiss cheese', 'goat cheese', 'cream cheese', 'blue cheese',
    'feta cheese', 'brie cheese', 'camembert cheese', 'ricotta cheese',
    'cottage cheese', 'mascarpone', 'heavy cream', 'cream', 'sour cream',
    'greek yogurt', 'full fat yogurt', 'kefir',
    
    # Vegetables - Leafy greens
    'lettuce', 'spinach', 'arugula', 'kale', 'swiss chard', 'collard greens',
    'romaine lettuce', 'iceberg lettuce', 'butter lettuce', 'endive',
    'radicchio', 'watercress', 'bok choy', 'cabbage', 'brussels sprouts',
    
    # Vegetables - Cruciferous
    'broccoli', 'cauliflower', 'cabbage', 'brussels sprouts', 'kohlrabi',
    'turnip', 'radish', 'daikon radish',
    
    # Vegetables - Other low-carb
    'asparagus', 'green beans', 'zucchini', 'yellow squash', 'spaghetti squash',
    'cucumber', 'celery', 'bell pepper', 'pepper', 'jalapeño', 'serrano pepper',
    'poblano pepper', 'mushroom', 'shiitake mushroom', 'portobello mushroom',
    'button mushroom', 'cremini mushroom', 'oyster mushroom',
    'eggplant', 'tomato', 'cherry tomato', 'onion', 'green onion', 'scallion',
    'leek', 'shallot', 'fennel', 'okra', 'artichoke', 'hearts of palm',
    'bamboo shoots', 'bean sprouts', 'jicama',
    
    # Fruits - Low carb
    'avocado', 'olive', 'olives', 'coconut', 'lemon', 'lime',
    'blackberry', 'raspberry', 'strawberry', 'cranberry',
    
    # Nuts & seeds
    'almond', 'walnut', 'pecan', 'macadamia', 'hazelnut', 'brazil nut',
    'pine nut', 'pistachio', 'pumpkin seed', 'sunflower seed',
    'chia seed', 'flax seed', 'hemp seed', 'sesame seed',
    'almond butter', 'peanut butter', 'sunflower butter', 'tahini',
    
    # Condiments & sauces
    'mayonnaise', 'mustard', 'dijon mustard', 'yellow mustard',
    'hot sauce', 'tabasco', 'sriracha', 'horseradish', 'wasabi',
    'vinegar', 'apple cider vinegar', 'white vinegar', 'red wine vinegar',
    'balsamic vinegar', 'rice vinegar', 'coconut aminos', 'tamari',
    'fish sauce', 'worcestershire sauce', 'pesto', 'tapenade',
    
    # Sweeteners
    'stevia', 'erythritol', 'monk fruit', 'xylitol', 'splenda',
    'sucralose', 'aspartame', 'allulose',
    
    # Baking & cooking
    'almond flour', 'coconut flour', 'psyllium husk', 'xanthan gum',
    'baking powder', 'baking soda', 'cream of tartar', 'gelatin',
    'agar agar', 'nutritional yeast',
    
    # Specialty keto products
    'mct powder', 'collagen powder', 'protein powder', 'electrolyte powder',
    'keto bread', 'shirataki noodles', 'kelp noodles', 'zucchini noodles',
    'cauliflower rice', 'pork rinds', 'beef jerky', 'sugar free jello',
    
    # Fermented foods
    'sauerkraut', 'kimchi', 'pickles', 'pickle', 'fermented vegetables',
    'kombucha', 'kefir water'
}

NON_KETO_KEYWORDS = { 
    # --- Grains, Flours & Starches ---
    'barley', 'bread', 'breadcrumb', 'cereal', 'corn', 'cornmeal', 'cornstarch', 'couscous', 
    'cracker', 'crouton', 'flour', 'granola', 'millet', 'oat', 'pasta', 'panko',
    'pretzel', 'quinoa', 'rice', 'rye', 'semolina', 'spelt', 'tapioca', 'tortilla', 'wheat',
    'baguette', 'bannock', 'ciabatta', 'noodle', 'rusk',

    # --- Starchy Vegetables & Tubers ---
    'parsnip', 'pea', 'plantain', 'potato', 'yam',

    # --- High-Sugar Fruits ---
    'banana', 'cherry', 'date', 'fig', 'grape', 'lychee', 'mango', 'pineapple', 
    'tangerine', 'apple', 'orange', 'strawberry', 'strawberrie', 
    
    # --- Alcoholic Beverages (High Carb) ---
    'vodka', 'rum', 'liqueur', 'beer', 'wine',  
    
    # --- Sugars & Syrups ---
    'agave', 'caramel', 'dextrose', 'fructose', 'glucose', 'honey', 
    'maltodextrin', 'maple', 'molasses', 'sugar', 'syrup',

    # --- Legumes ---
    'bean', 'chickpea', 'lentil', 'legume',

    # --- Processed Foods & Desserts ---
    'biscuit', 'cake', 'candy', 'chip', 'chocolate', 'cookie', 'dough',
    'doughnut', 'dumpling', 'ice cream', 'jam', 'jelly', 'ketchup', 'muffin', 'pie', 
    'pizza', 'popcorn', 'relish', 'sorbet', 'waffle',
}

# Database & Fuzzy Matching
# --- Tuning Parameters & Knowledge Bases ---
CONFIDENCE_THRESHOLD = 92
KETO_CARB_THRESHOLD_PER_100G = 10.0

# This block dynamically constructs the absolute path to the database file.
# It ensures the file can be found correctly, especially inside a Docker container.

# 1. Get the directory where THIS script (`diet_classifiers.py`) is located.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the absolute path to the data directory and the CSV file.
#    This assumes the `data` folder is located in the same directory as this script.
DB_PATH = os.path.join(SCRIPT_DIR, 'data', 'nutrition_database.csv')

# --- Global State for Database ---
NUTRITION_DF: Optional[pd.DataFrame] = None
FOOD_NAME_CHOICES: Optional[List[str]] = None
SEARCH_CACHE: Dict[str, Optional[Dict]] = {}


def _load_database() -> None:
    """
    Loads the nutritional database from the dynamically constructed absolute path.
    This function uses lazy loading and is robust to different execution environments.
    """
    global NUTRITION_DF, FOOD_NAME_CHOICES
    
    # Check if the database is already loaded into memory.
    if NUTRITION_DF is not None:
        return

    try:
        # The print statement now shows the full, unambiguous path it's trying to load.
        # This is excellent for debugging.
        print(f"INFO: Loading nutritional database from absolute path: '{DB_PATH}'...")
        NUTRITION_DF = pd.read_csv(DB_PATH)
        FOOD_NAME_CHOICES = NUTRITION_DF['food_name'].tolist()
        print("INFO: Database loaded successfully.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Nutritional database not found at '{DB_PATH}'.", file=sys.stderr)
        print("Please ensure the 'data' directory with 'nutrition_database.csv' is in the same directory as this script.", file=sys.stderr)
        # In a web app, sys.exit() can be too abrupt. A better approach for production
        # might be to raise an exception that the main app can catch.
        # For this task, exiting is acceptable.
        sys.exit(1)

def find_ingredient_nutrition(ingredient_name: str) -> Optional[Dict[str, Any]]:
    """
    Searches for an ingredient in the nutritional database.
    Returns high-carb dummy data for failed lookups to ensure conservative classification.
    """
    _load_database()
    if NUTRITION_DF is None or NUTRITION_DF.empty:
        return {'food_name': f"db_load_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        
    if ingredient_name in SEARCH_CACHE:
        return SEARCH_CACHE[ingredient_name]
        
    query_words = set(ingredient_name.split())
    if not query_words:
        result = {'food_name': f"empty_query: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        SEARCH_CACHE[ingredient_name] = result
        return result

    # Pre-filter choices
    filtered_choices = [name for name in FOOD_NAME_CHOICES if not query_words.isdisjoint(name.lower().split())]
    if not filtered_choices:
        if LOG_FAILED_LOOKUPS:
            with open("failed_lookups.log", "a") as f: 
                f.write(f"PREFILTER_FAIL: {ingredient_name}\n")
        result = {'food_name': f"prefilter_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        SEARCH_CACHE[ingredient_name] = result
        return result

    # Fuzzy string matching to find the best nutritional data match
    # This uses the fuzzywuzzy library to handle ingredient name variations

    # Step 1: Find the closest matching food name from our nutrition database
    # - ingredient_name: the cleaned/parsed ingredient (e.g., "tomato", "basil")
    # - filtered_choices: list of available food names in nutrition database
    # - scorer=fuzz.WRatio: uses Weighted Ratio algorithm for matching
    #   * WRatio is good for partial matches and handles word order differences
    #   * Example: "roma tomato" vs "tomato" would score highly
    # - extractOne(): returns the single best match as a tuple (match_string, confidence_score)
    best_match = process.extractOne(ingredient_name, filtered_choices, scorer=fuzz.WRatio)

    # Step 2: Validate the match quality before using it
    # best_match will be None if no matches found, or a tuple like ("tomato", 85)
    # best_match[1] is the confidence score (0-100 scale)
    # CONFIDENCE_THRESHOLD prevents accepting poor matches (typically 70-80)
    if best_match and best_match[1] >= CONFIDENCE_THRESHOLD:
        
        # Step 3: Extract the winning food name from the match result
        # best_match[0] contains the actual food name string from our database
        # This ensures we use the exact spelling that exists in our nutrition data
        matched_food_name = best_match[0]
        
        # Step 4: Look up nutritional information for the matched food
        # Filter the nutrition DataFrame to find rows where food_name matches exactly
        # .iloc[0] gets the first (and should be only) matching row
        # .to_dict() converts the pandas Series to a dictionary for easy access
        # Result contains all nutritional data: calories, protein, fat, vitamins, etc.
        result = NUTRITION_DF[NUTRITION_DF['food_name'] == matched_food_name].iloc[0].to_dict()
    else:
        # FIXED: Failed matches get HIGH carbs (not 0.0)
        if LOG_FAILED_LOOKUPS:
            with open("failed_lookups.log", "a") as f: 
                f.write(f"SCORE_FAIL: {ingredient_name}\n")
        result = {'food_name': f"score_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}

    SEARCH_CACHE[ingredient_name] = result
    return result

def is_ingredient_keto(ingredient: str, debug: bool = False) -> bool:
    """
    Enhanced keto checker with debug output and better logic.
    """
    clean_name = parse_ingredient(ingredient)
    
    if debug:
        print(f"  Checking: '{ingredient[:50]}...' -> cleaned: '{clean_name}'")
    
    # Empty ingredients are safe
    if not clean_name:
        if debug: print(f"    ✓ Empty ingredient - KETO")
        return True

    # Fast fail for known non-keto ingredients (CHECK THIS FIRST!)
    ingredient_words = set(clean_name.split())
    if not NON_KETO_KEYWORDS.isdisjoint(ingredient_words):
        matching_words = NON_KETO_KEYWORDS.intersection(ingredient_words)
        if debug: print(f"    ✗ Contains non-keto keyword(s): {matching_words} - NOT KETO")
        return False
        
    # Fast pass for known keto ingredients
    if not MANUAL_KETO_OVERRIDES.isdisjoint(ingredient_words):
        matching_words = MANUAL_KETO_OVERRIDES.intersection(ingredient_words)
        if debug: print(f"    ✓ Contains keto override(s): {matching_words} - KETO")
        return True




    # Database lookup for ambiguous ingredients
    nutrition_data = find_ingredient_nutrition(clean_name)
    if nutrition_data:
        carbs = nutrition_data.get('carbs', KETO_CARB_THRESHOLD_PER_100G + 1)
        is_keto_by_carbs = carbs <= KETO_CARB_THRESHOLD_PER_100G
        if debug:
            print(f"    Database: {nutrition_data['food_name']}, carbs: {carbs}g -> {'KETO' if is_keto_by_carbs else 'NOT KETO'}")
        return is_keto_by_carbs

    # Default to NOT KETO for unknown ingredients
    if debug: print(f"    ? Unknown ingredient, defaulting to NOT KETO")
    return False


def normalize_ingredient_input(data: Any) -> List[str]:
    """
    A robust parser that can handle multiple data formats and always returns a clean list of strings.
    - If input is a list, it returns it directly (for the web app).
    - If input is a string that looks like a list, it parses it (for the CSV).
    - If input is another type (like NaN), it returns an empty list.
    """
    if isinstance(data, list):
        # This handles the web app case where data is already a clean list.
        return data
        
    if not isinstance(data, str):
        # Handles other unexpected types from pandas like NaN
        return []

    # This handles the notebook/CSV case where data is a string
    try:
        # Try to parse it as a literal Python list.
        evaluated_list = ast.literal_eval(data)
        if isinstance(evaluated_list, list):
            return evaluated_list
    except (ValueError, SyntaxError):
        # If it fails, it might be a malformed string with no commas.
        return re.findall(r"'([^']*)'", data)
    
    # Fallback for completely unknown string formats
    return []

def is_keto(ingredients: Any) -> bool:
    """
    Production-ready entry point for Keto classification.
    It robustly handles different input data types.
    """
    # 1. Ensure we have a clean list of ingredients, regardless of the source.
    clean_ingredient_list = normalize_ingredient_input(ingredients)
    
    # 2. Apply the core per-ingredient logic.
    return all(map(is_ingredient_keto, clean_ingredient_list))

def is_vegan(ingredients: Any) -> bool:
    """
    Production-ready entry point for Vegan classification.
    It robustly handles different input data types.
    """
    # 1. Ensure we have a clean list of ingredients, regardless of the source.
    clean_ingredient_list = normalize_ingredient_input(ingredients)

    # 2. Apply the core per-ingredient logic.
    return all(map(is_ingredient_vegan, clean_ingredient_list))


def print_confusion_matrix(y_true, y_pred, classifier_type: str):
    """
    Calculates and prints a labeled confusion matrix for a given classifier type.

    Args:
        y_true: The true labels from the ground truth data.
        y_pred: The predicted labels from the classifier.
        classifier_type: A string, either 'keto' or 'vegan', to set the labels.
    """
    # First, ensure scikit-learn was imported successfully
    if 'confusion_matrix' not in globals():
        print("Warning: scikit-learn not found. Skipping confusion matrix.", file=sys.stderr)
        return

    try:
        # --- 1. Define labels based on the classifier type ---
        if classifier_type.lower() == 'keto':
            positive_class = 'Keto'
            negative_class = 'Non-Keto'
        elif classifier_type.lower() == 'vegan':
            positive_class = 'Vegan'
            negative_class = 'Non-Vegan'
        else:
            # A fallback for any other case
            positive_class = 'Positive'
            negative_class = 'Negative'

        # --- 2. Dynamically generate the labels for the DataFrame ---
        index_labels = [f'Actual: {negative_class}', f'Actual: {positive_class}']
        column_labels = [f'Predicted: {negative_class}', f'Predicted: {positive_class}']

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=index_labels, columns=column_labels)

        print(f"\n--- Confusion Matrix: {classifier_type.capitalize()} Classifier ---")
        print(cm_df)
        print("-" * 55)
        
        # Unpack the matrix values
        tn, fp, fn, tp = cm.ravel()
        
        # --- 3. Dynamically generate the descriptive printout ---
        print(f"  True Negatives (TN): {tn: >4} (Correctly predicted {negative_class})")
        print(f"  False Positives(FP): {fp: >4} (Incorrectly predicted {positive_class})")
        print(f"  False Negatives(FN): {fn: >4} (Incorrectly predicted {negative_class})")
        print(f"  True Positives (TP): {tp: >4} (Correctly predicted {positive_class})")
        print("-" * 55)
        
    except Exception as e:
        print(f"Could not generate confusion matrix for '{classifier_type}': {e}")

def main(args):
    """Main function to load data, run classifiers, and print a full report."""
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
   # ground_truth['ingredients'] = ground_truth['ingredients'].apply(parse_malformed_ingredient_string)
    

    _load_database()
    start_time = time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(is_keto)
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time()


    # --- UPDATED REPORTING SECTION ---
    print("\n" + "="*20 + " FINAL CLASSIFICATION REPORT " + "="*20)

    # --- Keto Report ---
    print("\n" + "="*20 + " Keto " + "="*20)
    print(classification_report(ground_truth['keto'], ground_truth['keto_pred']))
    # Call the new function with classifier_type='keto'
    print_confusion_matrix(ground_truth['keto'], ground_truth['keto_pred'], classifier_type='keto')

    # --- Vegan Report ---
    print("\n" + "="*20 + " Vegan " + "="*20)
    print(classification_report(ground_truth['vegan'], ground_truth['vegan_pred']))
    # Call the new function with classifier_type='vegan'
    print_confusion_matrix(ground_truth['vegan'], ground_truth['vegan_pred'], classifier_type='vegan')

    print(f"\n== Time taken: {end_time:.2f} seconds ==")
    
    return 0

if __name__ == "__main__":
    parser = ArgumentParser(description="Run diet classifiers on a ground truth dataset.")
    parser.add_argument("--ground_truth", type=str, default="/usr/src/data/ground_truth_sample.csv",
                        help="Path to the ground truth CSV file.")
    sys.exit(main(parser.parse_args()))